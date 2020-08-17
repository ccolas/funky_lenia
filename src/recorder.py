from src.params import *
import datetime, subprocess

class Recorder:
    RECORD_ROOT = 'record'
    FRAME_EXT = '.png'
    VIDEO_EXT = '.mov'
    GIF_EXT = '.gif'
    ANIM_FPS = 25
    ffmpeg_cmd = ['/usr/local/bin/ffmpeg',
                  '-loglevel', 'warning', '-y',  # glocal options
                  '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24',  # input options
                  '-s', '{}x{}'.format(SIZEX * PIXEL, SIZEY * PIXEL), '-r', str(ANIM_FPS),
                  '-i', '{input}',  # input pipe
                  # '-an', '-vcodec','h264', '-pix_fmt','yuv420p', '-crf','1',  # output options
                  '-an', '-vcodec', 'copy',  # output options
                  '{output}']  # ouput file

    def __init__(self, world):
        self.world = world
        self.is_recording = False
        self.is_save_frames = False
        self.record_id = None
        self.record_seq = None
        self.img_dir = None
        self.video_path = None
        self.video = None
        self.gif_path = None
        self.gif = None

    def toggle_recording(self, is_save_frames=False):
        self.is_save_frames = is_save_frames
        if not self.is_recording:
            self.start_record()
        else:
            self.finish_record()

    def start_record(self):
        global STATUS
        ''' https://trac.ffmpeg.org/wiki/Encode/H.264
            https://trac.ffmpeg.org/wiki/Slideshow '''
        self.is_recording = True
        STATUS.append("> start " + ("saving frames" if self.is_save_frames else "recording video") + " and GIF...")
        self.record_id = '{}-{}'.format(self.world.names[0].split('(')[0], datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
        self.record_seq = 1
        self.video_path = os.path.join(self.RECORD_ROOT, self.record_id + self.VIDEO_EXT)
        self.gif_path = os.path.join(self.RECORD_ROOT, self.record_id + self.GIF_EXT)
        self.img_dir = os.path.join(self.RECORD_ROOT, self.record_id)
        if self.is_save_frames:
            if not os.path.exists(self.img_dir):
                os.makedirs(self.img_dir)
        else:
            cmd = [s.replace('{input}', '-').replace('{output}', self.video_path) for s in self.ffmpeg_cmd]
            try:
                self.video = subprocess.Popen(cmd, stdin=subprocess.PIPE)  # stderr=subprocess.PIPE
            except FileNotFoundError:
                self.video = None
                STATUS.append("> no ffmpeg program found!")
        self.gif = []

    def save_image(self, img, filename=None):
        self.record_id = '{}-{}'.format(self.world.names[0].split('(')[0], datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
        img_path = filename + self.FRAME_EXT if filename else os.path.join(self.RECORD_ROOT, self.record_id + self.FRAME_EXT)
        img.save(img_path)

    def record_frame(self, img):
        if self.is_save_frames:
            img_path = os.path.join(self.RECORD_ROOT, self.record_id, '{:03d}'.format(self.record_seq) + self.FRAME_EXT)
            img.save(img_path)
        else:
            if self.video:
                img_rgb = img.convert('RGB').tobytes()
                self.video.stdin.write(img_rgb)
        self.gif.append(img)
        self.record_seq += 1

    def finish_record(self):
        global STATUS
        if self.is_save_frames:
            STATUS.append("> frames saved to '" + self.img_dir + "/*" + self.FRAME_EXT + "'")
            cmd = [s.replace('{input}', os.path.join(self.img_dir, '%03d' + self.FRAME_EXT)).replace('{output}', self.video_path) for s in self.ffmpeg_cmd]
            try:
                subprocess.call(cmd)
            except FileNotFoundError:
                self.video = None
                STATUS.append("> no ffmpeg program found!")
        else:
            if self.video:
                self.video.stdin.close()
                STATUS.append("> video saved to '" + self.video_path + "'")
        durations = [1000 // self.ANIM_FPS] * len(self.gif)
        durations[-1] *= 10
        self.gif[0].save(self.gif_path, format=self.GIF_EXT.lstrip('.'), save_all=True, append_images=self.gif[1:], loop=0, duration=durations)
        self.gif = None
        STATUS.append("> GIF saved to '" + self.gif_path + "'")
        self.is_recording = False

