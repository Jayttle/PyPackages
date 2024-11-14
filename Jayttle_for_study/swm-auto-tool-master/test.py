import sha
import sha.cv as cv
from sha.points import *

if __name__ == '__main__':
    # for image, name in [
    #     (img_swm, 'swm'),
    #     (img_box, 'box'),
    #     (img_cola_b, 'cola_b'),
    #     (img_cola_o, 'cola_o'),
    #     (img_digua, 'digua'),
    #     (img_shutiao, 'shutiao'),
    # ]:
    #     print(name)
    #     res = match_many_object_on_image(img_full_screen, image, draw_rect=True, output_name=name + '.png')
    #     print(f'there are {len(res)} {name} in the screen')
    cv.fast_screen_shot(POS_GUEST_1_LT, POS_GUEST_1_RB)
