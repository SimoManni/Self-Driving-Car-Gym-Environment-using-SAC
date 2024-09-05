import gym
from gym import spaces
import numpy as np
import pygame
from collections import deque

from AutonomousCar import AutonomousCar
from settings import *

class RacingEnv(gym.Env):
    def __init__(self):
        """
        Must define self.observation_space and self.action_space
        """
        # Action space: [throttle, brake, steering]
        self.max_throttle = 0.5
        self.max_brake = 0.5
        self.max_steering = 5.0

        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, -self.max_steering], dtype=np.float32),
            high=np.array([self.max_throttle, self.max_brake, self.max_steering], dtype=np.float32),
            dtype=np.float32
        )

        # Observation space
        # Last 3 observations [distances, speed, angle]
        low_bound = np.zeros(17, dtype=np.float32)
        high_bound = np.array([70., 78.44743, 98.99495, 126.1943, 157.03503, 157.03503, 144.3087, 140.0357,
                               144.3087, 156.52477, 156.52477, 126.1943, 98.99495, 78.26238, 70., V_MAX, 360.],
                              dtype=np.float32)
        self.observation_space = spaces.Box(
            low=low_bound,
            high=high_bound,
            dtype=np.float32
        )


        self.car = AutonomousCar()


        self.contour_points = BARRIERS

        if N_STATES > 1:
            self.prev_states = deque(maxlen=N_STATES-1)

        self.VIS_BARRIERS = True
        self.VIS_CHECKPOINTS = True
        self.VIS_PERCEPTION = True
        self.car.VIS_PERCEPTION = self.VIS_PERCEPTION

        track = pygame.image.load('track.png')
        self.image = pygame.transform.scale(track, (WIDTH, HEIGHT))

    def reset(self):
        """
        Returns the observation of the initial state
        Resets the environment to initial state so that a new episode (independent of previous ones) can start
        """
        self.car.reset()
        init_state = self._get_state()

        self.prev_states.clear()
        self.prev_states.appendleft(init_state)
        self.prev_states.appendleft(init_state)

        self.checkpoint_index = 0

        self.screen = None
        self.agent = None

        return np.tile(init_state, 3)

    def step(self, action):
        """
        Returns: next observation, reward, done and optionally additional info
        """
        done = False
        self.car.update(action)
        reward = 0
        passed_checkpoints = 0

        # Penalty for using throttle and brake at the same time
        if action[0] != 0 and action[1] != 0:
            reward -= 0.01 / (np.abs(action[0] - action[1]) + 0.01)

        # Penalty for steering when the car is very slow or still
        # if self.car.speed < 0.5:
        #     reward -= np.abs(action[2]) * (1 - self.car.speed)

        # Check if car passes checkpoint
        if self.car.checkpoint():
            reward += 10
            passed_checkpoints += 1

        # Check if collision occurred
        if self.car.check_collision():
            reward += -50
            done = True

        # Reward for completing lap or reaching end or part
        if passed_checkpoints == len(self.checkpoints):
            reward += 100
            done = True

        current_observation = self._get_state()
        observation = np.concatenate((current_observation, np.array(self.prev_states).flatten()))
        self.prev_states.appendleft(current_observation)
        return observation, reward, done

    def render(self, agent=None, time_limit=True):
        """
        Returns: None
        Show the current environment state e.g the graphical window.
        """

        MAX_TIME_SECONDS = 10

        # Initialize Pygame
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption(f'Autonomous Car - SAC')

        running = True
        start_time = pygame.time.get_ticks()
        observation = self.reset()
        score = 0
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            # Check elapsed time
            if time_limit:
                current_time = pygame.time.get_ticks()
                elapsed_time_seconds = (current_time - start_time) / 1000.0
                if elapsed_time_seconds >= MAX_TIME_SECONDS:
                    running = False  # Exit the loop if maximum time exceeded

            action = agent.choose_action(observation)
            observation, reward, done = self.step(action)
            score += reward
            if done:
                self.car.reset()
                score = 0

            self._draw(screen)
            self._write_info(screen, action, score)
            pygame.display.update()

        # Quit Pygame
        pygame.quit()

    def close(self):
        """
        Returns: None
        Cleanup all resources (graphical windows, threads) etc.
        """
        raise NotImplementedError

    # Additional functions
    def _get_state(self):
        # Normalize distances
        max_distances = self.observation_space.high[:-2]
        distances = self.car.perceive() / max_distances
        return np.concatenate((distances, [self.car.speed], [self.car.angle / 360]))

    def _get_starting_position(self, index=None):
        # Definition of starting points and angles
        x1 = self.checkpoints[:, 0]
        y1 = self.checkpoints[:, 1]
        x2 = self.checkpoints[:, 2]
        y2 = self.checkpoints[:, 3]
        x_middle = (x1 + x2) / 2
        y_middle = (y1 + y2) / 2

        # Definition of starting points
        starting_points = []
        for i, (x_m, y_m) in enumerate(zip(x_middle, y_middle)):
            index = (i + 1) % (len(x_middle))
            x = (x_m + x_middle[index]) / 2
            y = (y_m + y_middle[index]) / 2
            starting_points.append([x, y])

        STARTING_POINTS = np.roll(np.array(starting_points).astype(int), 1, axis=0)

        # Definition of starting angles
        angles = []
        for i, (x_m, y_m) in enumerate(zip(x_middle, y_middle)):
            index = (i + 1) % (len(x_middle))
            angle = np.arctan2(y_middle[index] - y_m, x_middle[index] - x_m) * 180 / np.pi
            corrected_angle = (270 - angle % 360) if angle % 360 < 270 else (angle % 360)
            angles.append(corrected_angle)

        STARTING_ANGLES = np.roll(np.array(angles).astype(int), 1)

        # Wrap-around of checkpoints and definition of starting configuration
        if index != 0:
            self.checkpoints = np.vstack((self.checkpoints[index:], self.checkpoints[:index]))
        elif index == 0:
            self.checkpoints = self.checkpoints
        return STARTING_POINTS[index], STARTING_ANGLES[index]

    # Functions for visualization
    def _draw(self, screen):
        screen.blit(self.image, (0, 0))
        screen.blit(self.image, (0, 0))
        text = pygame.font.Font(None, 30).render(f"Laps completed: {self.car.laps}", True, (0, 0, 0))
        if self.VIS_BARRIERS:
            self._draw_lines(screen)
        if self.VIS_CHECKPOINTS:
            self._draw_checkpoints(screen)

        self.car.draw(screen)

    def _draw_lines(self, screen):
        pygame.draw.lines(screen, (255, 0, 0), True, self.contour_points[0], 5)
        pygame.draw.lines(screen, (255, 0, 0), True, self.contour_points[1], 5)
        for point in np.vstack(self.contour_points):
            pygame.draw.circle(screen, (0, 0, 255), point, 5)

    def _draw_checkpoints(self, screen):
        for idx, line in enumerate(self.car.checkpoints):
            if idx < self.car.passed_checkpoints:
                pygame.draw.line(screen, (255, 255, 0), line[:2], line[2:], 3)
            else:
                pygame.draw.line(screen, (255, 0, 0), line[:2], line[2:], 3)


    def _write_info(self, screen, action, score):
        font = pygame.font.Font(None, 30)
        text_state = font.render(f'Speed: {self.car.speed:.1f}, Angle: {self.car.angle:.1f}',
                                 True,
                                 (0, 0, 0))
        screen.blit(text_state, (20, 20))
        text_input = font.render(f'Throttle: {action[0]:.1f}, Brake: {action[1]:.1f}, Steering: {action[2]:.1f}',
                                 True,
                                 (0, 0, 0))
        screen.blit(text_input, (20, 50))
        text_score = font.render(f'Score: {score:.1f}',
                                 True,
                                 (0, 0, 0))
        screen.blit(text_score, (20, 80))
