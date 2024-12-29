import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_file(filename):
    img=cv2.imread(filename)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # plt.imshow(img)
    # plt.show()
    return img

filename="image.png"
img= read_file(filename)

org_img=np.copy(img)

def edge_mask(img, line_size, blur_value):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_blur=cv2.medianBlur(gray,blur_value)

    edges=cv2.adaptiveThreshold(gray_blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,line_size,blur_value)

    return edges

edges=edge_mask(img,7,9)



def color_quantization(img, k):
    #transform the image
    data=np.float32(img).reshape((-1,3))

    #determine criteria
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)

    #apply k-means
    ret, label, center=cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    #convert back to 8-bit values
    center=np.uint8(center)
    result=center[label.flatten()]

    #reshape to the original image dimensions
    result=center[label.flatten()]
    result=result.reshape(img.shape)
    return result

img=color_quantization(img, k=9)
# plt.imshow(img)
# plt.show()

blurred=cv2.bilateralFilter(img,d=7,sigmaColor=200,sigmaSpace=200)

# plt.imshow(blurred)
# plt.show()

def cartoon():
    c=cv2.bitwise_and(blurred,blurred,mask=edges)
    plt.imshow(c)
    plt.show()

    plt.imshow(org_img)
    plt.show()
    
cartoon()
