class Settings:
     """存储游戏《外星人入侵》中所有设置的类"""
     def __init__(self) -> None:
          self.screen_width = 1200
          self.screen_height = 800
          self.bg_color = (230, 230, 230)
          self.ship_speed = 1.5

          # 子弹设置
          self.bullet_speed = 2.5
          self.bullet_width = 3
          self.bullet_height = 15
          self.bullet_color = (60, 60, 60)
          self.bullets_allowed = 3

          self.alien_speed = 1.0
          self.fleet_drop_speed = 10
          # fleet_direction 为 1 表示向右移动，为-1 表示向左移动
          self.fleet_direction = 1

          # 飞船设置
          self.ship_speed = 1.5
          self.ship_limit = 3

          self.speedup_scale = 1.1

          # 外星人分数的提高速度
          self.score_scale = 1.5
          self.initialize_dynamic_settings()

     def initialize_dynamic_settings(self):
          """初始化随游戏进行而变化的设置"""
          self.ship_speed = 1.5
          self.bullet_speed = 2.5
          self.alien_speed = 1.0

          self.alien_points = 50
          # fleet_direction 为 1 表示向右，为-1 表示向左
          self.fleet_direction = 1

     def increase_speed(self):
          """提高速度设置的值"""
          self.ship_speed *= self.speedup_scale
          self.bullet_speed *= self.speedup_scale
          self.alien_speed *= self.speedup_scale

          self.alien_points = int(self.alien_points * self.score_scale)
          print(self.alien_points)