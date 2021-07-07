import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torchvision.utils import save_image
import pdb


# Input: images (W,H,3,D,D), pytorch tensors, values should be between 0 and 1
def create_image_from_subimages(images, file_name):
    cur_shape = images.shape
    images = images.permute(1, 0, 2, 3, 4).contiguous()  # (H,W,3,D,D)
    images = images.view(-1, *cur_shape[-3:])  # (H*W, 3, D, D)
    save_image(images, fp=file_name, nrow=cur_shape[0])
    #Note: Weird that it is cur_shape[0] but cur_shape[1] produces incorrect image


# Input: imgs (H,W,D,D,3) or (H,W,3,D,D), numpy arrays, values ints between 0-225
def plot_multi_image(imgs, save_dir, caption=None):
    if imgs.shape[-1] != 3:
        imgs = np.moveaxis(imgs, 2, -1)
    if imgs.dtype is not np.uint8:
        imgs = np.clip((imgs * 255), 0, 255).astype(np.uint8)

    rows, cols, imsize, imsize, _ = imgs.shape

    fig = plt.figure(figsize=(6*cols, 6*rows))
    ax = []
    count = 1
    for i in range(rows):
        for j in range(cols):
            ax.append(fig.add_subplot(rows, cols, count))
            ax[-1].set_yticklabels([])
            ax[-1].set_xticklabels([])
            if caption is not None:
                ax[-1].set_title('%0.4f' % caption[i][j])
            plt.imshow(imgs[i, j])
            count += 1

    plt.savefig(save_dir)
    plt.close('all')


# Input: aTensor (B,***) -> (H,W,***)
def reshapeToRowsCols(aTensor, h, w):
    lastShapes = aTensor.shape[1:]
    return aTensor.reshape([h, w] + list(lastShapes))


# Input: Plotting a confusion matrix
# https://stackoverflow.com/questions/2148543/how-to-write-a-confusion-matrix-in-python/30385488
def plot_confusion_matrix(pred_vals, true_vals, normalize=False, filename="confusion.png",
                          title='Confusion matrix', cmap=plt.cm.gray_r):
    conf_mat = confusion_matrix(true_vals, pred_vals)
    if normalize:
        conf_mat = conf_mat / conf_mat.sum(axis=1)
    num_classes = max(np.max(pred_vals), np.max(true_vals))

    plt.matshow(conf_mat, cmap=cmap)  # imshow
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, tick_marks) #, rotation=45)
    plt.yticks(tick_marks, tick_marks)
    # plt.ylabel(range(num_classes))
    # plt.xlabel(range(num_classes))
    plt.savefig(filename)
    plt.close('all')


# Testing functions
def test_confusion_matrix():
    pred_vals = np.random.randint(0, 10, 1000)
    true_vals = np.random.randint(0, 10, 1000)
    plot_confusion_matrix(pred_vals, true_vals)


if __name__ == "__main__":
    # timeDataloader()
    test_confusion_matrix()
