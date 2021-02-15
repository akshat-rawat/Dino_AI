# used NEAT algorithm

import pygame
import os
import neat
import math

pygame.font.init()

WIN_WIDTH = 1000
WIN_HEIGHT = 550
FLOOR = 430
STAT_FONT = pygame.font.SysFont("arial", 50)
END_FONT = pygame.font.SysFont("arial", 70)

clock = pygame.time.Clock()

win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
gen = 0

cactus_img = pygame.transform.scale2x(pygame.image.load(os.path.join("images", "cactus1.png")).convert_alpha())
dino_images = [pygame.transform.scale2x(pygame.image.load(os.path.join("images", "dino" + str(x) + ".png"))) for x in
               range(1, 3)]
base_img = pygame.transform.scale2x(pygame.image.load(os.path.join("images", "base.png")).convert_alpha())
jump_img = pygame.transform.scale2x(pygame.image.load(os.path.join("images", "dinoJump.png")).convert_alpha())


class Dino:
    IMGS = dino_images
    jumpList = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1,
                -1, -1, -1, -1, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3,
                -3, -3, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4]

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.height = 64
        self.width = 64
        self.vel = 0
        self.isJump = False
        self.jumpCount = 0
        self.img_count = 0
        self.runCount = 0
        self.img = self.IMGS[0]
        self.hitbox = pygame.Rect(self.x + 4, self.y, self.width + 20, self.height + 30)

    def draw(self, win):
        if self.isJump:
            self.y -= self.jumpList[self.jumpCount] * 1.3
            win.blit(jump_img, (self.x, self.y))
            self.jumpCount += 1
            if self.jumpCount > 108:
                self.jumpCount = 0
                self.isJump = False
                self.runCount = 0
        else:
            if self.runCount >= 40:
                self.runCount = 0
            win.blit(self.IMGS[int(self.runCount // 20)], (self.x, self.y))
            self.runCount += 1 + self.vel

        self.hitbox = pygame.Rect(self.x + 20, self.y, self.width + 5, self.height + 30)


class Cactus:
    VEL = 5

    def __init__(self, x):
        self.x = x
        self.y = 350
        self.height = 64
        self.width = 64
        self.bottom = x
        self.img = cactus_img
        self.passed = False
        self.hitbox = pygame.Rect(self.x, self.y, self.width - 20, self.height + 20)

    def move(self):
        self.x -= self.VEL

    def draw(self, win):
        win.blit(self.img, (self.x, self.y))
        self.hitbox = pygame.Rect(self.x, self.y, self.width - 20, self.height + 20)


class Base:
    WIDTH = base_img.get_width()
    VEL = 5
    IMG = base_img

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        self.x1 -= self.VEL
        self.x2 -= self.VEL
        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH

        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))


def draw_window(dinos, cacti, base, score, gen):
    win.fill((0, 0, 0))
    base.draw()
    for cactus in cacti:
        cactus.draw(win)

    for dino in dinos:
        dino.draw(win)

    score_label = STAT_FONT.render("Score: " + str(score), 1, (255, 255, 255))
    win.blit(score_label, (WIN_WIDTH - score_label.get_width() - 15, 10))

    score_label = STAT_FONT.render("Gens: " + str(gen - 1), 1, (255, 255, 255))
    win.blit(score_label, (10, 10))

    score_label = STAT_FONT.render("Alive: " + str(len(dinos)), 1, (255, 255, 255))
    win.blit(score_label, (10, 50))

    pygame.display.update()


def eval_genomes(genomes, config):
    global win, gen
    gen += 1

    nets = []
    dinos = []
    ge = []

    # generate genomes for creating multiple dinosaurs and networks
    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        dinos.append(Dino(230, 350))
        ge.append(genome)

    base = Base(FLOOR)
    cacti = [Cactus(1300)]
    score = 0
    score_count = 0

    clock = pygame.time.Clock()

    run = True
    while run and len(dinos) > 0:
        clock.tick(150)
        score_count += 1
        if score_count > 40:
            score = score + 1
            score_count = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
                break

        cactus_ind = 0
        cacti[cactus_ind].mask = pygame.mask.from_surface(cacti[cactus_ind].img)
        re = cacti[cactus_ind].mask.get_bounding_rects()

        # initialize every genome with fitness of 0.1
        for x, dino in enumerate(dinos):
            ge[x].fitness += 0.1

            input1 = dino.x + dino.width
            input2 = math.hypot(((dino.x + dino.width) - cacti[cactus_ind].x),
                                (dino.y - (cacti[cactus_ind].bottom - re[0].height)))
            input3 = math.hypot(((dino.x + dino.width) - cacti[cactus_ind].x),
                                (dino.y - (cacti[cactus_ind].y + cacti[cactus_ind].height)))
            input4 = math.hypot(((dino.x + dino.width) - (cacti[cactus_ind].x + re[0].width)),
                                (dino.y - cacti[cactus_ind].bottom))

            # provide various inputs to generate a network
            output = nets[dinos.index(dino)].activate((input1, input2, input3, input4))

            # use tanh function to determine the jump of dino
            if output[0] > 0.5:
                dino.isJump = True

        base.move()

        rem = []
        add_cactus = False
        for cactus in cacti:
            cactus.move()

            # remove genomes, dinosaurs that got hit by cactus and decrease their fitness by 1
            for dino in dinos:
                if dino.hitbox.colliderect(cactus.hitbox):
                    ge[dinos.index(dino)].fitness -= 1
                    nets.pop(dinos.index(dino))
                    ge.pop(dinos.index(dino))
                    dinos.pop(dinos.index(dino))

            if cactus.x + cactus.width < 0:
                rem.append(cactus)

            if not cactus.passed and cactus.x < dino.x:
                cactus.passed = True
                add_cactus = True

        if add_cactus:
            for genome in ge:
                genome.fitness += 5                         # add +5 fitness of every genome that passes a cactus
            cacti.append(Cactus(WIN_WIDTH))

        for r in rem:
            cacti.remove(r)

        draw_window(dinos, cacti, base, score, gen)


def run(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(eval_genomes, 50)

    print('\nBest genome:\n{!s}'.format(winner))                # print stats after every generation


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config_feedforward.txt')
    run(config_path)
