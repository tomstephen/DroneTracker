import cv2
import numpy as np
import os
import time

pathToTemplates = "queryImages"
threshold = 0.8

templates = []

# load template images and append to "templates" array
for image_path in os.listdir(pathToTemplates):
    input_path = os.path.join(pathToTemplates, image_path)
    templates.append(cv2.imread(input_path,0)) # read template image in greyscale

def highlightMatches(frame, templates, threshold): # find & highlight drone in given frame
    # template scales in %
    scale_min = 50
    scale_max = 200
    scale_step = 10

    img_rgb = frame
    img_grey = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    img_out = frame

    for template in templates:
        for scale_percentage in xrange(scale_min, scale_max, scale_step):
            scale_factor = float(scale_percentage) / 100
            scaled_template = cv2.resize(template, None, fx=scale_factor, fy=scale_factor) # scale template

            w, h = scaled_template.shape[::-1]
            res = cv2.matchTemplate(img_grey,scaled_template,cv2.TM_CCOEFF_NORMED) # find matches with template
            loc = np.where(res >= threshold)

            for pt in zip(*loc[::-1]): # highlight (rectangle around) matches
                cv2.rectangle(img_out, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

    return img_out

def separateFrames(): # separate video into frames
    print("Separating Frames...")
    vidcap = cv2.VideoCapture("Video1.mp4")
    success,image = vidcap.read()
    count = 0
    success = True
    while success:
        cv2.imwrite("frame%d.png" % count, image)     # save frame as PNG file
        success,image = vidcap.read()
        count += 1
    return count

def recombineFrames(frameCount): # combine frames into video
    print("Combining Frames...")
    init = cv2.imread("frame1.png")
    height, width, layers = init.shape

    video = cv2.VideoWriter("video.avi",-1,30,(width,height))

    for i in range(0, frameCount):
        frame = cv2.imread("frame%d.png" % i) # load PNG file
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()

def timeRemaining(frameCount, currentFrameIndex, processingTimes):
    average_time = sum(processingTimes) / float(len(processingTimes))
    frames_remaining = frameCount - currentFrameIndex
    remaining_time = average_time * frames_remaining
    return remaining_time

def editFrames(frameCount): # loop through frames and pass to "highlightMatches" function
    print("Editing Frames...")

    times = []

    for i in range(0,frameCount):
        start_time = time.time()

        frame = cv2.imread("frame%d.png" % i) # load PNG file
        highlighted = highlightMatches(frame, templates, threshold)
        cv2.imwrite("frame%d.png" % i, highlighted)

        end_time = time.time()

        time_diff = end_time - start_time

        times.append(time_diff)

        print("Frame " + str(i) + " done in %.2fs" % time_diff)
        print("Approx. %.2fs remaining." % timeRemaining(frameCount, i, times))

def deleteFrames(frameCount): # delete all left over frames
    print("Deleting Frames...")
    for i in range(0,frameCount):
        os.remove("frame%d.png" % i)

program_start_time = time.time()

frameCount = separateFrames()
print("There are " + str(frameCount) + " frames.")

editFrames(frameCount)
recombineFrames(frameCount)
deleteFrames(frameCount)

program_end_time = time.time()
program_diff_time = program_end_time - program_start_time
print("Program completed in %.2fs" % program_diff_time)
