import gym
from gym import spaces
import numpy as np
import pygame
from collections import deque

from autonomous_car import AutonomousCar
from settings import *

class RacingEnv(gym.Env):
    def __init__(self, multi_agent=True):
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

        self.SIM = multi_agent

        if self.SIM:
            self.cars = []
            for key in list(CONFIGURATIONS.keys()):
                car = AutonomousCar(key)
                self.cars.append(car)

            if N_STATES > 1:
                self.deques = []
                for _ in range(len(self.cars)):
                    self.deques.append(deque(maxlen=N_STATES-1))
        else:
            self.car = AutonomousCar('normal')
            if N_STATES > 1:
                self.prev_states = deque(maxlen=N_STATES-1)

        self.contour_points = BARRIERS

        self.VIS_BARRIERS = True
        self.VIS_CHECKPOINTS = True
        self.VIS_PERCEPTION = True

        if self.SIM:
            for car in self.cars:
                car.VIS_PERCEPTION = self.VIS_PERCEPTION
        else:
            self.car.VIS_PERCEPTION = self.VIS_PERCEPTION

        track = pygame.image.load('images/track.png')
        self.image = pygame.transform.scale(track, (WIDTH, HEIGHT))

    def reset(self):
        """
        Returns the observation of the initial state
        Resets the environment to initial state so that a new episode (independent of previous ones) can start
        """
        if self.SIM:
            init_states = []
            for i, car in enumerate(self.cars):
                car.reset()
                init_state = self._get_state(car)
                init_states.append(np.tile(init_state, 3))
                self.deques[i].clear()
                self.deques[i].appendleft(init_state)
                self.deques[i].appendleft(init_state)

            return np.array(init_states)
        else:
            self.car.reset()
            init_state = self._get_state(self.car)

            self.prev_states.clear()
            self.prev_states.appendleft(init_state)
            self.prev_states.appendleft(init_state)

            return np.tile(init_state, 3)

    def step(self, actions):
        """
        Returns: next observation, reward, done and optionally additional info
        """
        if self.SIM:
            new_state_list = []
            reward_list = []
            done_list = []

            for car, action, prev_states in zip(self.cars, actions, self.deques):
                done = False
                if car.active:
                    car.update(action)
                    reward = 0

                    # Penalty for using throttle and brake at the same time
                    if action[0] != 0 and action[1] != 0:
                        reward -= 0.01 / (np.abs(action[0] - action[1]) + 0.01)

                    if car.config != 'straight1' and car.config != 'straight2':
                        # Penalty for steering when the car is very slow or still
                        if car.speed < 0.5:
                            reward -= 0.1 * np.abs(action[2]) * (1 - car.speed)

                    # Check if car passes checkpoint
                    if car.checkpoint():
                        reward += 10

                    # Check if collision occurred
                    if car.check_collision():
                        reward += -50
                        done = True

                    current_observation = self._get_state(car)
                    observation = np.concatenate((current_observation, np.array(prev_states).flatten()))
                    prev_states.appendleft(current_observation)

                    new_state_list.append(observation)
                    reward_list.append(reward)
                    done_list.append(done)
                else:
                    reward = np.nan
                if done:
                    car.active = False
                    continue

            return np.array(new_state_list), np.array(reward_list), np.array(done_list)
        else:
            action = actions
            done = False
            self.car.update(action)
            reward = 0

            # Penalty for using throttle and brake at the same time
            if action[0] != 0 and action[1] != 0:
                reward -= 0.01 / (np.abs(action[0] - action[1]) + 0.01)

            # Penalty for steering when the car is very slow or still
            if self.car.speed < 0.5:
                reward -= 0.1 * np.abs(action[2]) * (1 - self.car.speed)

            # Check if car passes checkpoint
            if self.car.checkpoint():
                reward += 10

            # Check if collision occurred
            if self.car.check_collision():
                reward += -50
                done = True

            current_observation = self._get_state(self.car)
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
    def _get_state(self, car):
        # Normalize distances
        max_distances = self.observation_space.high[:-2]
        distances = car.perceive() / max_distances
        return np.concatenate((distances, [car.speed], [car.angle / 360]))

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
