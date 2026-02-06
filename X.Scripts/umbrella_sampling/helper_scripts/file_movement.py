import os

os.mkdir('frames')
# max set arbitarily to cover all possible generated configurations, however for longer simulation a better way is probably needed
for i in range(0,1000):
        file_name = f"frame-{i}_run_frame.sh"
        if os.path.exists(file_name):
            os.chdir('./frames')
            os.mkdir(f'frame-{i}')
            os.system(f"mv ../frame-{i}_run_frame.sh ./frame-{i}/")
            
            os.chdir('../')
