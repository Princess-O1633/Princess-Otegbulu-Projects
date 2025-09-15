import cv2 as cv

img = cv.imread('cat_large.jpg') 
capture = cv.VideoCapture("dog.mp4")

def rescale_frame(frame, scale=.25):
    # Images, Videos, Live Videos
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def change_res(width, height):
    # Live video
    capture.set(3,width)
    capture.set(4,height)

while True:
    isTrue, frame = capture.read()

    frame_resized = rescale_frame(frame)
    cv.imshow('Video', frame)  # display each frame
    cv.imshow('Video Resized', frame_resized)

    if cv.waitKey(20) & 0xFF == ord('d'):  # If the letter d pressed
        break


resized_image = rescale_frame(img)
cv.imshow('cat', img)
cv.imshow('Image', resized_image)


capture.release()
cv.waitKey(0)
cv.destroyAllWindows()