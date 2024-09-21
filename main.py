import pygame
import os
import math
import sys
import neat
import matplotlib.pyplot as plt

SCREEN_WIDTH =  1244
SCREEN_HEIGHT = 1016
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

TRACK = pygame.image.load("C:\\Users\\bigfi\\OneDrive\\Desktop\\DriveAI-master\\Assets\\track3.png")
CAR =   pygame.image.load("C:\\Users\\bigfi\\OneDrive\\Desktop\\DriveAI-master\\Assets\\car.png")

class Car(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.original_image = CAR
        self.image = self.original_image
        self.rect = self.image.get_rect(center=(490, 820))
        self.vel_vector = pygame.math.Vector2(0.8, 0)
        self.angle = 0
        self.rotation_vel = 3
        self.direction = 0
        self.alive = True
        self.radars = []

    def update(self):
        self.radars.clear()
        self.drive()
        self.rotate()
        for radar_angle in (-45, 0, 45):  # Radarı 3'e indiriyoruz
            self.radar(radar_angle)
        self.collision()
        self.data()

    def drive(self):
        self.rect.center += self.vel_vector * 2

    def collision(self):
        length = 40
        collision_point_right = [int(self.rect.center[0] + math.cos(math.radians(self.angle + 18)) * length),
                                 int(self.rect.center[1] - math.sin(math.radians(self.angle + 18)) * length)]
        collision_point_left = [int(self.rect.center[0] + math.cos(math.radians(self.angle - 18)) * length),
                                int(self.rect.center[1] - math.sin(math.radians(self.angle - 18)) * length)]

        if SCREEN.get_at(collision_point_right) == pygame.Color(2, 105, 31, 255) \
                or SCREEN.get_at(collision_point_left) == pygame.Color(2, 105, 31, 255):
            self.alive = False

        pygame.draw.circle(SCREEN, (0, 255, 255, 0), collision_point_right, 4)
        pygame.draw.circle(SCREEN, (0, 255, 255, 0), collision_point_left, 4)

    def rotate(self):
        if self.direction == 1:
            self.angle -= self.rotation_vel
            self.vel_vector.rotate_ip(self.rotation_vel)
        if self.direction == -1:
            self.angle += self.rotation_vel
            self.vel_vector.rotate_ip(-self.rotation_vel)

        self.image = pygame.transform.rotozoom(self.original_image, self.angle, 0.1)
        self.rect = self.image.get_rect(center=self.rect.center)

    def radar(self, radar_angle):
        length = 0
        x = int(self.rect.center[0])
        y = int(self.rect.center[1])

        while not SCREEN.get_at((x, y)) == pygame.Color(2, 105, 31, 255) and length < 200:
            length += 1
            x = int(self.rect.center[0] + math.cos(math.radians(self.angle + radar_angle)) * length)
            y = int(self.rect.center[1] - math.sin(math.radians(self.angle + radar_angle)) * length)

        pygame.draw.line(SCREEN, (255, 255, 255, 255), self.rect.center, (x, y), 1)
        pygame.draw.circle(SCREEN, (0, 255, 0, 0), (x, y), 3)

        dist = int(math.sqrt(math.pow(self.rect.center[0] - x, 2)
                             + math.pow(self.rect.center[1] - y, 2)))

        self.radars.append([radar_angle, dist])

    def data(self):
        input = [0, 0, 0]  # Radarı 3'e indiriyoruz, girişi de 3 yapıyoruz
        for i, radar in enumerate(self.radars):
            input[i] = int(radar[1])
        return input

def remove(index):
    cars.pop(index)
    ge.pop(index)
    nets.pop(index)

def eval_genomes(genomes, config):
    global cars, ge, nets

    cars = []
    ge = []
    nets = []

    for genome_id, genome in genomes:
        cars.append(pygame.sprite.GroupSingle(Car()))
        ge.append(genome)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        genome.fitness = 0

    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        SCREEN.blit(TRACK, (0, 0))

        if len(cars) == 0:
            break

        for i, car in enumerate(cars):
            ge[i].fitness += 1
            if not car.sprite.alive:
                remove(i)

        for i, car in enumerate(cars):
            output = nets[i].activate(car.sprite.data())
            if output[0] > 0.7:
                car.sprite.direction = 1
            if output[1] > 0.7:
                car.sprite.direction = -1
            if output[0] <= 0.7 and output[1] <= 0.7:
                car.sprite.direction = 0

        for car in cars:
            car.draw(SCREEN)
            car.update()
        pygame.display.update()

stats = None
generations = []
fitness_scores = []

def run(config_path):
    global pop, stats, generations, fitness_scores

    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    def update_plot():
        # Clear the current figure
        plt.clf()
        plt.xlabel("Jenerasyon")
        plt.ylabel("Fitness")
        plt.title("NEAT Araba AI - Jenerasyonlar Üzerinden Fitness")
        plt.grid(True)
        
        # Update plot with new data
        generations.append(len(generations) + 1)
        fitness_scores.append(max((genome.fitness or 0) for genome_id, genome in pop.population.items() if genome.fitness is not None))
        
        plt.plot(generations, fitness_scores, marker='o', color='b')
        plt.draw()
        plt.pause(0.1)  # Short pause to update the plot

    # Run the NEAT algorithm and update plot every generation
    def run_and_plot():
        for generation in range(50):  # Number of generations
            pop.run(eval_genomes, 1)  # Run for a single generation
            update_plot()  # Update the plot after each generation

    plt.figure(figsize=(10, 5))
    run_and_plot()

    # Keep the plot window open
    plt.show(block=True)

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    run(config_path)