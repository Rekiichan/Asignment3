import numpy as np
from scipy.signal import butter, lfilter, freqz
from matplotlib import pyplot as plt

def get_histogram(gray_img):
    rows,cols = gray_img.shape
    gray_img = np.array(gray_img).astype(int)
    freq = np.zeros(256).astype(int)
    for r in range(rows):
        for c in range(cols):
            freq[gray_img[r][c]] += 1
    return freq

def Median_Filter(Gray_Matrix,ksize,Smax):
    rows,cols = Gray_Matrix.shape
    Filtered_Image = np.zeros([rows,cols])
    h = (Smax - 1)//2
    Padded_Image = np.pad(Gray_Matrix,(h,h),mode='reflect')
    for r in range(rows):
        for c in range(cols):
            k = ksize
            K_Image = Padded_Image[r:r+k,c:c+k]
            while True:
                A1 = np.median(K_Image)
                A2 = np.median(K_Image)
                if A1 > 0 and A2 < 0:
                    B1 = int(Gray_Matrix[r,c]) - int(np.min(K_Image))
                    B2 = int(Gray_Matrix[r,c]) - int(np.min(K_Image))
                    if B1 > 0 and B2 < 0:
                        Filtered_Image[r,c] = Gray_Matrix[r,c]
                    else:
                        Filtered_Image[r,c] = np.median(K_Image)
                    break
                else:
                    k += 1
                    Snew = k*2+1
                    if Snew <= Smax:
                        K_Image = Padded_Image[r:r+k,c:c+k]
                    else:
                        Filtered_Image[r,c] = np.median(K_Image)
                        break
    return Filtered_Image

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


if __name__ == "__main__":
    Image = np.loadtxt('4_1.asc')
    order = 5
    fs = 30.0
    cutoff = 3.667

    Median_Filtered_Image = Median_Filter(Image,7,11)
    Filtered_Image = butter_lowpass_filter(Median_Filtered_Image, cutoff, fs, order)

    Hist_ORI = get_histogram(Image)
    Hist_Medium_filter = get_histogram(Filtered_Image)
    Hist_LP_Med_filter = get_histogram(Filtered_Image)

    fig = plt.figure(figsize=(10, 7)) #create figure
    
    fig.add_subplot(2, 3, 1)
    plt.imshow(Image)
    plt.title('Original Image')
    plt.gray()
    
    fig.add_subplot(2, 3, 2)
    plt.imshow(Median_Filtered_Image)
    plt.title('Medium Filter')
    plt.gray()
    
    fig.add_subplot(2, 3, 3)
    plt.imshow(Median_Filtered_Image)
    plt.title('Low-pass + Medium Filter')
    plt.gray()
    
    fig.add_subplot(2, 3, 4)
    axis_X = np.arange(0, 256, 1)
    plt.bar(axis_X,Hist_ORI)
    plt.title('Histogram')
    
    fig.add_subplot(2, 3, 5)
    axis_X = np.arange(0, 256, 1)
    plt.bar(axis_X,Hist_Medium_filter)
    plt.title('Histogram')
    
    fig.add_subplot(2, 3, 6)
    axis_X = np.arange(0, 256, 1)
    plt.bar(axis_X,Hist_LP_Med_filter)
    plt.title('Histogram')
    
    plt.show()