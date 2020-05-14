import nibabel as nib
import os
import numpy as np
import shutil
from PIL import Image
from tqdm import tqdm
import cv2


def to_standard(img):
    img[np.isnan(img)] = 0
    for k in range(np.shape(img)[2]):
        st = img[:, :, k]
        if np.amin(st) != np.amax(st):
            st -= np.amin(st)
            st /= np.amax(st)
        st *= 255
    return img


def to_2d(data="train_for_SRNN"):
    folders = os.listdir(data)
    for folder in folders:
        if '2d' not in folder:
            count = 0
            print("Current Folder: {}".format(folder))
            folder_path = os.path.join(data, folder)

            save_path = folder_path + '_2d'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            else:
                shutil.rmtree(save_path)
                os.makedirs(save_path)

            files = os.listdir(folder_path)
            for file in files:
                file_name = file[0: file.find('.')]
                file_path = os.path.join(folder_path, file)
                label = to_standard(nib.load(file_path).get_fdata())
                print(label.shape)

                slices = label.shape[2]
                for s in range(slices):
                    position = s / slices
                    new_name = os.path.join(save_path, (str(s) + '_{:.1f}_' + file_name + '_lab.png').format(position))

                    slice_temp = label[:, :, s]
                    slice_img = Image.fromarray(alter_intensity(slice_temp.round().astype(np.uint8), folder))

                    slice_img.save(new_name, "PNG", dpi=[300, 300], quality=95)
                    count += 1
                    print("Finishing processing: {}".format(new_name))

            with open("num_record.txt", 'a') as f:
                f.write(folder + ': ' + str(count // 2) + '\n')


def verify_intensity(data="train_for_SRNN"):
    folders = os.listdir(data)
    for folder in folders:
        if '2d' in folder:
            folder_path = os.path.join(data, folder)
            files = os.listdir(folder_path)
            for file in files:
                file_path = os.path.join(folder_path, file)
                lab = Image.open(file_path)
                lab_array = np.array(lab)
                intensity = np.unique(lab_array)
                print(intensity)


def alter_intensity(img, folder):
    if 'seg' in folder:
        img[img == 51] = 200
        img[img == 153] = 88
        img[img == 255] = 244
    else:
        img[img == 85] = 88
        img[img == 170] = 200
        img[img == 255] = 244
    return img


def unite_folders(data="train_for_SRNN"):
    save = "in_use"
    if not os.path.exists(save):
        os.makedirs(save)
    else:
        shutil.rmtree(save)
        os.makedirs(save)

    folders = os.listdir(data)
    for folder in folders:
        if '2d' in folder:
            folder_path = os.path.join(data, folder)
            files = os.listdir(folder_path)
            for file in files:
                file_path = os.path.join(folder_path, file)
                save_path = os.path.join(save, file)
                shutil.copy(file_path, save_path)
    print("Done.")


def verify_shape(data="crop"):
    files = os.listdir(data)
    save_regular = []
    save_strange = []
    center_strange = []
    for file in files:
        file_path = os.path.join(data, file)
        lab = np.array(Image.open(file_path))
        if 'gt' not in file:
            save_regular.append(lab.shape)
        else:
            save_strange.append(lab.shape)
            center_strange.append(compute_center(lab))
    print(save_regular)
    print(save_strange)
    # print(center_strange)


def compute_center(label):
    label = np.expand_dims(label, axis=-1)
    points = np.where(label > 0)
    return np.array([[np.average(points[0][points[2] == j]), np.average(points[1][points[2] == j])] for j in
                     range(label.shape[-1])])


# old method, don't use it
def resize_and_crop(data='in_use'):
    files = os.listdir(data)
    save_path = "resize"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        shutil.rmtree(save_path)
        os.makedirs(save_path)

    for file in files:
        file_path = os.path.join(data, file)
        lab = Image.open(file_path)
        if 'gt' in file:
            pass
            height, width = lab.size
            lab_new = lab.resize((int(2.5*height), int(2.5*width)), Image.NEAREST)
            print(lab_new.size)
            print(np.unique(lab_new))
            lab_new.save("resize/" + file)
        else:
            shutil.copy(file_path, os.path.join(save_path, file))
            lab_new = np.lib.pad(lab, 60, 'constant', constant_values=0)
            Image.fromarray(lab_new).save("resize/" + file)


# old method, don't use it
def center_crop_2d(data='resize', center_roi=(120, 120, 1)):

    files = os.listdir(data)
    save_path = "crop"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        shutil.rmtree(save_path)
        os.makedirs(save_path)

    for file in files:

        file_path = os.path.join(data, file)
        label = np.expand_dims(Image.open(file_path), axis=-1)

        if 'gt' in file:
            assert np.all(np.array(center_roi) <= np.array(label.shape)), print(
                'Patch size exceeds dimensions.')

            center = compute_center(label)
            where_are_nan = np.isnan(center)
            center[where_are_nan] = int(label.shape[0] // 2)

            x = np.array([center[i][0] for i in range(label.shape[-1])]).astype(np.int)
            y = np.array([center[i][1] for i in range(label.shape[-1])]).astype(np.int)

            x = x[0]
            y = y[0]

            beginx = x - center_roi[0]
            beginy = y - center_roi[1]
            endx = x + center_roi[0]
            endy = y + center_roi[1]

            gt = label[beginx:endx, beginy:endy, :]
            Image.fromarray(np.squeeze(gt)).save(os.path.join(save_path, file))

        else:
            gt = label
            Image.fromarray(np.squeeze(gt)).save(os.path.join(save_path, file))


def center_crop_xz(data='in_use', roi=(120, 120)):

    files = os.listdir(data)
    save_path = "cropped_in_use"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        shutil.rmtree(save_path)
        os.makedirs(save_path)

    for file in tqdm(files):
        file_path = os.path.join(data, file)
        label = np.array(Image.open(file_path))

        if 'gt' in file:
            assert np.all(np.array(roi) <= np.array(label.shape)), print(
                'Patch size exceeds dimensions.')

            center = compute_center(label)
            where_are_nan = np.isnan(center)
            center[where_are_nan] = int(label.shape[0] // 2)
            center_2d = np.array(center[0], dtype=np.int32)
            window_size = roi[0] * 2

            begin = np.where(center_2d - window_size // 2 < 0,
                             0,
                             center_2d - window_size // 2)

            end = np.where(center_2d - window_size // 2 + window_size > label.shape[:],
                           label.shape[:],
                           center_2d - window_size // 2 + window_size)

            offset1 = np.where(center_2d - window_size // 2 < 0,
                               window_size // 2 - center_2d,
                               0)

            offset2 = np.where(center_2d - window_size // 2 + window_size > label.shape[:],
                               center_2d - window_size // 2 + window_size - label.shape[:],
                               0)

            label_crop = label[begin[0]:end[0], begin[1]:end[1]]

            label_pad = np.pad(label_crop, pad_width=((offset1[0], offset2[0]),
                                                      (offset1[1], offset2[1])),
                               mode='constant')
            print(label_pad.shape)

            Image.fromarray(np.squeeze(label_pad)).save(os.path.join(save_path, file))

        else:
            label_pad = np.pad(label, 60, 'constant', constant_values=0)
            print(label_pad.shape)
            Image.fromarray(np.squeeze(label_pad)).save(os.path.join(save_path, file))


def center_crop_old_data(data_path='../dataset/train_2d',
                         save_path="../dataset/train_2d_crop",
                         roi=(120, 120)):
    print("Currently crop ROI for dataset: ", data_path)
    files = os.listdir(data_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        shutil.rmtree(save_path)
        os.makedirs(save_path)

    for file in files:
        if 'lab' in file:
            label_path = os.path.join(data_path, file)
            label = np.array(Image.open(label_path))
            image_name = file.replace("_lab.png", "_img.png")
            image = np.array(Image.open(os.path.join(data_path, image_name)))

            assert np.all(np.array(roi) <= np.array(label.shape)), print(
                'Patch size exceeds dimensions.')

            center = compute_center(label)
            where_are_nan = np.isnan(center)
            center[where_are_nan] = int(label.shape[0] // 2)
            center_2d = np.array(center[0], dtype=np.int32)
            window_size = roi[0] * 2

            begin = np.where(center_2d - window_size // 2 < 0,
                             0,
                             center_2d - window_size // 2)

            end = np.where(center_2d - window_size // 2 + window_size > label.shape[:],
                           label.shape[:],
                           center_2d - window_size // 2 + window_size)

            offset1 = np.where(center_2d - window_size // 2 < 0,
                               window_size // 2 - center_2d,
                               0)

            offset2 = np.where(center_2d - window_size // 2 + window_size > label.shape[:],
                               center_2d - window_size // 2 + window_size - label.shape[:],
                               0)

            label_crop = label[begin[0]:end[0], begin[1]:end[1]]
            image_crop = image[begin[0]:end[0], begin[1]:end[1]]

            label_pad = np.pad(label_crop, pad_width=((offset1[0], offset2[0]),
                                                      (offset1[1], offset2[1])),
                               mode='constant')
            image_pad = np.pad(image_crop, pad_width=((offset1[0], offset2[0]),
                                                      (offset1[1], offset2[1])),
                               mode='constant')
            # print(label_pad.shape)
            # print(image_pad.shape)

            Image.fromarray(np.squeeze(label_pad)).save(os.path.join(save_path, file))
            Image.fromarray(np.squeeze(image_pad)).save(os.path.join(save_path, image_name))
            print("Done: ", image_name)


def center_crop_old_old(data_path='../dataset/train_2d',
                         save_path="../dataset/train_2d_crop",
                         roi=(120, 120, 1)):
    print("Currently crop ROI for dataset: ", data_path)
    files = os.listdir(data_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        shutil.rmtree(save_path)
        os.makedirs(save_path)

    for file in files:
        if 'lab' in file:
            label_path = os.path.join(data_path, file)
            label = np.array(Image.open(label_path))
            image_name = file.replace("_lab.png", "_img.png")
            image = np.array(Image.open(os.path.join(data_path, image_name)))
            label = np.expand_dims(label, axis=-1)
            image = np.expand_dims(image, axis=-1)

            assert np.all(np.array(roi) <= np.array(label.shape)), print(
                'Patch size exceeds dimensions.')
            center = compute_center(label)
            where_are_nan = np.isnan(center)
            center[where_are_nan] = int(label.shape[0] // 2)

            x = np.array([center[i][0] for i in range(label.shape[-1])]).astype(np.int)
            y = np.array([center[i][1] for i in range(label.shape[-1])]).astype(np.int)

            x = x[0]
            y = y[0]

            beginx = x - roi[0]
            beginy = y - roi[1]
            endx = x + roi[0]
            endy = y + roi[1]

            gt = label[beginx:endx, beginy:endy, :]
            img = image[beginx:endx, beginy:endy, :]

            Image.fromarray(np.squeeze(gt)).save(os.path.join(save_path, file))
            Image.fromarray(np.squeeze(img)).save(os.path.join(save_path, image_name))
            print("Done: ", image_name)


def check_shape(data="test_crop"):

    files = os.listdir(data)
    for file in files:
        file_path = os.path.join(data, file)
        lab = Image.open(file_path)
        # if 'gt' not in file:
        print("Shape of current label: ", lab.size)
        print("Intensities of current label: ", np.unique(lab))


class LabelErosion:

    def __init__(self, data_path="cropped_in_use", save_path="noise_in_use", debug=False):
        self.data_path = data_path
        self.save_path = save_path
        self.debug = debug
        self.intensities = self.get_intensities()

    def change_intensity(self, img_raw):
        img = np.zeros_like(img_raw)
        for i in range(len(self.intensities)):
            if i < len(self.intensities) - 1:
                img[img_raw == self.intensities[i]] = self.intensities[i+1]
            else:
                img[img_raw == self.intensities[i]] = self.intensities[0]
        return img

    def get_intensities(self):
        images = os.listdir(self.data_path)
        image_path = os.path.join(self.data_path, images[0])
        img = cv2.imread(image_path, 0)
        intensities = np.unique(img)
        return intensities

    def sample_class(self, array):
        return array[np.random.randint(0, 3)]

    def produce_noise(self, image):
        image_path = os.path.join(self.data_path, image)
        img = cv2.imread(image_path, 0)

        img_1 = self.change_intensity(img)
        img_2 = self.change_intensity(img_1)
        img_3 = self.change_intensity(img_2)

        candidate = np.stack([img_1, img_2, img_3], axis=-1)
        selection = np.random.uniform(0, 1, img.shape)

        sampled_img = np.apply_along_axis(self.sample_class, axis=-1, arr=candidate)
        # print(sampled_img.shape)

        result = np.where(selection < 0.1, sampled_img, img)

        if self.debug:
            cv2.imshow("img_raw", img)
            cv2.imshow("img_1", img_1)
            cv2.imshow("img_2", img_2)
            cv2.imshow("img_3", img_3)
            cv2.imshow("sampled", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return result

    def main(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        else:
            shutil.rmtree(self.save_path)
            os.makedirs(self.save_path)

        images = os.listdir(self.data_path)
        for image in images:
            noise_image = self.produce_noise(image)
            image = "noise_" + image
            cv2.imwrite(os.path.join(self.save_path, image), noise_image)
            print("Finish process: ", image)


if __name__ == '__main__':

    # to_2d()
    # verify_intensity()
    # unite_folders()
    # verify_shape()
    # resize_and_crop()
    # center_crop_2d()
    # center_crop_xz()
    # check_shape()

    # center_crop_old_data(data_path="test/test_img", save_path="test/test_save_1")
    # center_crop_old_data()
    # center_crop_old_old(data_path="test/test_img", save_path="test/test_save")
    center_crop_old_old()

    # L = LabelErosion(data_path="../dataset/train_2d_crop", save_path="../dataset/noise_train_2d_crop")
    # L.main()





