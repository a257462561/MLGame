"""The template of the main script of the machine learning process
"""

import games.arkanoid.communication as comm
from games.arkanoid.communication import ( \
    SceneInfo, GameInstruction, GameStatus, PlatformAction
)

def ml_loop():
    """The main loop of the machine learning process
    This loop is run in a separate process, and communicates with the game process.
    Note that the game process won't wait for the ml process to generate the
    GameInstruction. It is possible that the frame of the GameInstruction
    is behind of the current frame in the game process. Try to decrease the fps
    to avoid this situation.
    """

    # === Here is the execution order of the loop === #
    # 1. Put the initialization code here.
    ball_position_history = [ ]
    # 2. Inform the game process that ml process is ready before start the loop.
    comm.ml_ready()
    # 3. Start an endless loop.
    while True:
        
        # 3.1. Receive the scene information sent from the game process.
        scene_info = comm.get_scene_info()
        ball_position_history.append(scene_info.ball)
        platform_center_x = scene_info.platform[0] + 20
        if (len(ball_position_history)) == 1 :
            ball_going_down = 0
        elif ball_position_history [-1][1] - ball_position_history [-2][1] > 0:
            ball_going_down = 1
            vy = ball_position_history[-1][1] - ball_position_history[-2][1]
            vx = ball_position_history[-1][0] - ball_position_history[-2][0]
        else:
            ball_going_down = 0
        # 3.2. If the game is over or passed, the game process will reset
        #      the scene and wait for ml process doing resetting job.
        if scene_info.status == GameStatus.GAME_OVER or \
            scene_info.status == GameStatus.GAME_PASS:
            # Do some stuff if needed

            # 3.2.1. Inform the game process that ml process is ready
            comm.ml_ready()
            continue

        # 3.3. Put the code here to handle the scene information
        if len(scene_info.bricks) > 15 :
            if ball_going_down == 1 and ball_position_history[-1][1] >= 0:
                ball_destination = ball_position_history[-1][0] + ((395-ball_position_history[-1][1])/vy)*vx
                if ball_destination >= platform_center_x:
                    ball_destination = 195-(ball_destination-195)
                    comm.send_instruction(scene_info.frame, PlatformAction.MOVE_RIGHT)
                elif ball_destination <= platform_center_x:
                    ball_destination = -ball_destination
                    comm.send_instruction(scene_info.frame, PlatformAction.MOVE_LEFT)
                
            else :
                if platform_center_x < 100:
                    comm.send_instruction(scene_info.frame, PlatformAction.MOVE_RIGHT)
                elif platform_center_x > 100:
                    comm.send_instruction(scene_info.frame, PlatformAction.MOVE_LEFT)
                else:
                    comm.send_instruction(scene_info.frame, PlatformAction.NONE)
        else:
            
            if ball_going_down == 1 and ball_position_history[-1][1] >= 200:
                ball_destination = ball_position_history[-1][0] + ((395-ball_position_history[-1][1])/vy)*vx
                if ball_destination >= platform_center_x:
                    ball_destination = 195-(ball_destination-200)
                    comm.send_instruction(scene_info.frame, PlatformAction.MOVE_RIGHT)
                elif ball_destination <= platform_center_x:
                    ball_destination = -ball_destination
                    comm.send_instruction(scene_info.frame, PlatformAction.MOVE_LEFT)
              
            else :
                if platform_center_x < 80:
                    comm.send_instruction(scene_info.frame, PlatformAction.MOVE_RIGHT)
                elif platform_center_x > 80:
                    comm.send_instruction(scene_info.frame, PlatformAction.MOVE_LEFT)
                else:
                    comm.send_instruction(scene_info.frame, PlatformAction.NONE)