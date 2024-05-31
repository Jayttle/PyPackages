import re
import time
import numpy as np
import pandas as pd
import jieba
import jieba.posseg as pseg
from PIL import Image
from wordcloud import WordCloud
import seaborn as sns
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt
from matplotlib import font_manager as fm
from tqdm import tqdm
from paddlenlp import Taskflow


# sns.set_theme(style="ticks")
# font = "/usr/share/fonts/winfont/simsun.ttc"
# fp = fm.FontProperties(fname=font)
# plt.rcParams["axes.unicode_minus"] = False

