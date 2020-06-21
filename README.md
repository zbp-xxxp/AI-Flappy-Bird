# AI-Flappy-Bird
使用PARL 1.3.1 版本复现Flappy-Bird游戏,结合DQN算法,让小鸟尽可能地拿高分

我一开始在本地用可视化界面跑,速度比较慢,后来比较稳定后,我把项目搬到了AI Studio上,项目地址如下:
https://aistudio.baidu.com/aistudio/projectdetail/580622

因为在服务器上跑的时候不需要展示可视化界面,所以训练速度比较快,另外,需要对游戏环境做小小的调整,在flappy_bird/game/BirdEnv.py文件里添加如下代码:
import os 
os.environ["SDL_VIDEODRIVER"] = "dummy"
否则会报错 ==> pygame.error: No available video device

原项目大概在3300个episode之后,模型收敛
我在复现时没有调整超参数,实际运行时,在3300个episode之后,total_reward最高为32.80,meanReward为8.08,效果还不是很好,这里暂时不做优化,只调通程序
