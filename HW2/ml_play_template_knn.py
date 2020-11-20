import games.arkanoid.communication as comm
from games.arkanoid.communication import ( \
    SceneInfo, GameInstruction, GameStatus, PlatformAction
)

import numpy as np
import pickle
def ml_loop():
    """The main loop of the machine learning process"""


    # Here is the execution order of the loop === #
    ball_position_history=[]
    filename='C:\\Users\\user\\Desktop\\MLGame\\games\\arkanoid\\knn.sav'
    model=pickle.load(open(filename, 'rb'))
    comm.ml_ready()

    # Start an endless loop.
    while True:
        scene_info = comm.get_scene_info()
        ball_position_history.append(scene_info.ball)
        if len(ball_position_history) > 2:
            inp_temp=np.array([ball_position_history[-2][0], ball_position_history[-2][1], ball_position_history[-1][0], ball_position_history[-1][1], scene_info.platform[0]])
            input=inp_temp[np.newaxis, :]
            print(input)
        
            if scene_info.status == GameStatus.GAME_OVER or \
                scene_info.status == GameStatus.GAME_PASS:
                comm.ml_ready()
                continue
        
            move=model.predict(input)
            print(move)
            if move > 0:
                comm.send_instruction(scene_info.frame, PlatformAction.MOVE_RIGHT)
            elif move < 0:
                comm.send_instruction(scene_info.frame, PlatformAction.MOVE_LEFT)
            else:
                comm.send_instruction(scene_info.frame, PlatformAction.NONE)