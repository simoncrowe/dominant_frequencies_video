from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from colorsys import rgb_to_hls, hls_to_rgb
import numpy as np
import time
import pygame.mixer as mixer

TESTING = False
VIDEO_FILE_PATH = "test-clip.mp4"

class VideoFileFreqAnalyser():
    def __init__(self, filepath, sample_rate = 44100):
        video = VideoFileClip(filepath, audio = False) # No mask, no audio
        self.fps = video.fps
        self.height = video.h
        self.width = video.w
        self.sample_rate = sample_rate
        audio_array = AudioFileClip(filepath, fps=sample_rate).to_soundarray()
        self.audio = self.mix_sterio(audio_array) # may retain sterio if desired
        frame_count = int((len(self.audio) / sample_rate) * self.fps) + 1
        self.video = np.empty((frame_count), dtype=object)
        frame_number = 0
        print('Loading video...')
        for f in video.iter_frames():
            self.video[frame_number] = f
            frame_number += 1
        print('Finished loading video ' + str(len(self.video))+' frames!')


    def mix_sterio(self, sterio_channels):
        mono_shape = (len(sterio_channels))
        mono_channel = np.empty(mono_shape, dtype = np.float)
        for i in range(len(mono_channel)):
            mono_channel[i] = sterio_channels[i][0] + sterio_channels[i][1]
        return mono_channel

    def get_hues(self, 
                frame, 
                sample_spacing, 
                horizontal_range, 
                vertical_range):
        """
        Returns spaced samples of hue values for given ranges of a 
        two dimensional numpy pixel array.
        horizontal_range and vertical_range are two-element tuples
        """
        sample_count = (int((horizontal_range[1] - horizontal_range[0])
                        /sample_spacing) *
                        int((vertical_range[1] - vertical_range[0])
                        /sample_spacing))
        hues = np.empty((sample_count), dtype = float)
        hue_index = 0
        if TESTING:
            print('Max vertical = ' 
                    + str(int(vertical_range[1]/sample_spacing) - 1))
            print('Max horizontal = ' 
                    + str(int(horizontal_range[1]/ sample_spacing)))
        for y in range(int(vertical_range[0]/sample_spacing), 
                       (int(vertical_range[1]/sample_spacing))):
            #print('Y = ' + str(y))
            for x in range(int(horizontal_range[0]/ sample_spacing),
                           (int(horizontal_range[1]/ sample_spacing) - 1)):
                #print('X = ' + str(x))
                r = float(frame[y*sample_spacing, x*sample_spacing][0]) / 255.0
                g = float(frame[y*sample_spacing, x*sample_spacing][1]) / 255.0
                b = float(frame[y*sample_spacing, x*sample_spacing][2]) / 255.0
                #print(frame[y*sample_spacing][x*sample_spacing][0])
                #print(r)
                #print(int((rgb_to_hls(r, g, b))[0] * 360.0))
                hues[hue_index] = rgb_to_hls(r, g, b)[0]


                #if hues[hue_index] < 0:
                #    print('Bad hue value at ' + str(y*sample_spacing) + ', ' 
                #        + str(x*sample_spacing))
                #    print(hues[hue_index])
                #print(hues[pixel_index])
                hue_index += 1
        return hues
        
    def find_modal_hue(self, hues, hue_granularity):
        hue_band_values = [0] * hue_granularity
        hue_band_thresholds = [0] * hue_granularity
        hue_counts = [0] * hue_granularity
        half_band_width = (1.0/hue_granularity)/2
        # special setting for band zero
        hue_band_thresholds[0] = -1.0 # to avoid >= comparson
        hue_band_values[0] = half_band_width
        i = 1 # skip zero index when looping
        while i < hue_granularity:
            threshold = i / hue_granularity
            hue_band_thresholds[i] = threshold
            hue_band_values[i] = threshold + half_band_width
            i += 1
        bad_hue_count = 0
        for h in hues:
            matched = False
            hue_band = hue_granularity # set to highest band
            while not matched:
                hue_band -= 1
                try:
                    if h > hue_band_thresholds[hue_band]:
                        #print('Hue ' + str(h) + 
                        #' is greater than band threshold ' 
                        #+ str(hue_band_thresholds[hue_band]))
                        hue_counts[hue_band] += 1
                        matched = True
                except TypeError:
                    if TESTING:
                        print('Skipping bad hue: ' + str(h))
                    matched = True # LIES!
                    bad_hue_count += 1
                except IndexError:
                    if TESTING:
                        print('Skipping bad hue: ' + str(h))
                    bad_hue_count += 1   
                    matched = True # LIES!
                    
        dominant_hue_index = 0
        greatest_hue_count = 0
        for i in range(len(hue_counts)):
            if (hue_counts[i] > greatest_hue_count):
                greatest_hue_count = hue_counts[i]
                #print('Hue index ' + str(i) + 'is greatest:' + str(greatest_hue_count))
                dominant_hue_index = i
                
        return float(hue_band_values[dominant_hue_index])
        
        
    def get_dominant_hue(self,
                        frame_number,
                        sample_spacing = 32,
                        hue_granularity = 256,
                        segments= (1, 1)):
        """
        For the frame numbered frame_number, return the dominant hue as
        a floating point number.
        sample_spacing sets the speed/quality of the function.
            With a sample_spacing value of 1, every pixel would be sampled;
            with 16, only every 16*16th pixel is sampled - this is quicker.
        hue_granularity specifies how many bands of hue
            the video frame is searched for.
            The default is 256, matching a typical 8 bit chennel.
        """
        frame = self.video[frame_number]
        pixel_index = 0
        
        frame_hue_segments = np.empty(segments, dtype=object)
        segments_modal_hues = np.empty(segments, dtype=float)
        for h in range(segments[0]):
            for v in range(segments[1]):
                if TESTING:
                    print('Getting hue values for segment [' + str(h) 
                        + ',' + str(v) + '] at :' + str(time.time()) + '...')
                frame_hue_segments[h,v] = self.get_hues(  
                    frame, sample_spacing,
                    (int(self.width/segments[0]) * h,
                     int(self.width/segments[0]) * (h + 1)),
                    (int(self.height/segments[1]) * v, 
                     int(self.height/segments[1]) * (v + 1)))
                if TESTING:
                    print('Finished getting hue values for segment [' 
                        + str(h)  + ',' + str(v) + '] at :' + str(time.time()))   
                if TESTING:
                    print('Finding modal hues for segment [' + str(h) 
                        + ',' + str(v) + '] at :' + str(time.time()) + '...')
                segments_modal_hues[h,v] = self.find_modal_hue(
                                                    frame_hue_segments[h,v],
                                                    hue_granularity)
                if TESTING:
                    print('Finished finding modal hues for segment [' + str(h) 
                        + ',' + str(v) + '] at :' + str(time.time()))   
        return segments_modal_hues

    def get_dominant_frequency(self, frame_number):
        """
        Returns the dominant audio frequency of a given frame in hertz.
        """
        samples_per_frame = int(self.sample_rate / self.fps)
        frame_samples = self.audio[frame_number *  samples_per_frame : 
                                   (frame_number + 1) * samples_per_frame]
        if TESTING:
            print('Starting FFT processing at time: ' + str(time.time()))
        w = np.fft.fft(frame_samples)
        freqs = np.fft.fftfreq(len(w))
        #print(freqs.min(), freqs.max())
        # (-0.5, 0.499975)
        # Find the peak in the coefficients
        idx = np.argmax(np.abs(w))
        freq = freqs[idx]
        freq_in_hertz = abs(freq * self.sample_rate)
        if TESTING:
            print('Finishing FFT processing at time: ' + str(time.time()))
        return freq_in_hertz
    
def generate_sine_wave(frequency, duration, volume=0.5, sample_rate=44100):
    """
    Creates a sine wave array of a given frequency 
    that can be pplayed by pygame.mixer
    Returns numpy array of 16 bit integers of a set duration, 
    based on the sample rate provided.
    """    
   
    samples_float = ((np.sin(2*np.pi*np.arange(sample_rate*duration)
                        * frequency/sample_rate)).astype(np.float32))
    samples_int = np.empty((len(samples_float)), dtype=np.int16)
    
    normalisation_multiplier = int(32768 * volume)
    for si in range(len(samples_float)):
        samples_int[si] = int(samples_float[si] * normalisation_multiplier)
    return samples_int

# DEMO CODE:
avfreq = VideoFileFreqAnalyser(VIDEO_FILE_PATH, 44100)
mixer.init( frequency = 44100,
            channels = 1)

print('Started at: ' + str(time.time()))

for i in range(2244):
    f = avfreq.get_dominant_frequency(i)
    hue  = avfreq.get_dominant_hue(i, 28, 256, (2, 1))
    sine_wave = generate_sine_wave(f, 1.0/avfreq.fps, float(i/2244))
    sine_sound = mixer.Sound(array=sine_wave)
    sine_sound.play()
    print('FRAME ' + str(i) + ':')
    print('Hues  = ')
    print(hue)
    print('Freq = ' + str(f) + ' Hz\n')
    
print('Ended at: ' + str(time.time()))

