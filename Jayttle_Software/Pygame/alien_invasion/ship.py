import pygame
from pygame.sprite import Sprite

class Ship(Sprite):
    """管理飞船的类"""
    def __init__(self, ai_game) -> None:
        """初始化飞船并设置其初始位置"""
        super().__init__()
        self.screen = ai_game.screen
        self.settings = ai_game.settings
        self.screen_rect = ai_game.screen.get_rect()

        # 加载飞船图像并获取其外接矩形
        self.image = pygame.image.load(r'D:\Program Files (x86)\Software\OneDrive\PyPackages\Jayttle_Software\Pygame\images\ship.bmp')
        self.rect = self.image.get_rect()

        # 在飞船的属性 x 中存储一个浮点数
        self.x = float(self.rect.x)
        # 每艘新飞船都放在屏幕底部的中央
        self.rect.midbottom = self.screen_rect.midbottom
        
        # 移动标志（飞船一开始不移动）
        self.moving_right = False
        self.moving_left = False


    def update(self):
        """根据移动标志调整飞船的位置"""
        # 更新飞船而不是 rect 对象的 x 值
        if self.moving_right and self.rect.right < self.screen_rect.right:
            self.x += self.settings.ship_speed
        if self.moving_left and self.rect.left > 0:
            self.x -= self.settings.ship_speed
        self.rect.x = self.x

        
    def blitme(self):
        """在指定位置绘制飞船"""
        self.screen.blit(self.image, self.rect)

    def center_ship(self):
        """将飞船放在屏幕底部的中央"""
        self.rect.midbottom = self.screen_rect.midbottom
        self.x = float(self.rect.x)