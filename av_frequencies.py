from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from colorsys import rgb_to_hls, hls_to_rgb
import numpy as np
from scipy.io.wavfile import read as wavread
import time

TESTING = False


class VideoFileFreqAnalyser():
    def __init__(self, video_path, audio_path):
        self.video = VideoFileClip(video_path, audio = False)
        audio_array = wavread(audio_path)
        self.audio = audio_array[1]
        self.sample_rate = audio_array[0]
        #self.audio = self.mix_sterio(audio_arrayana) # may retain sterio if desired
        self.frame_count_appox = int(((len(self.audio)/self.sample_rate) 
                                                    * self.video.fps))



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
        hues = []
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
                hue = rgb_to_hls(r, g, b)[0]
                if hue != None:
                    hues.append(hue)


                #if hues[hue_index] < 0:
                #    print('Bad hue value at ' + str(y*sample_spacing) + ', ' 
                #        + str(x*sample_spacing))
                #    print(hues[hue_index])
                #print(hues[pixel_index])
                #hue_index += 1
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
            for hue_band in range(hue_granularity - 1, -1, -1): # Back iterate
                try:
                    if h > hue_band_thresholds[hue_band]:
                        hue_counts[hue_band] += 1
                        break
                except TypeError:
                    if TESTING:
                        print('Skipping bad hue: ' + str(h))
                    bad_hue_count += 1
                except IndexError:
                    if TESTING:
                        print('Skipping bad hue: ' + str(h))
                    bad_hue_count += 1
        print(str(bad_hue_count) + ' bad hue values found!')
        
        dominant_hue_index = 0
        greatest_hue_count = 0
        for i in range(len(hue_counts)):
            if (hue_counts[i] > greatest_hue_count):
                greatest_hue_count = hue_counts[i]
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
        frame_time = frame_number / self.video.fps
        frame = self.video.get_frame(frame_time)
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
                    (int(self.video.w/segments[0]) * h,
                     int(self.video.w/segments[0]) * (h + 1)),
                    (int(self.video.h/segments[1]) * v, 
                     int(self.video.h/segments[1]) * (v + 1)))
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
        samples_per_frame = int(self.sample_rate / self.video.fps)
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

