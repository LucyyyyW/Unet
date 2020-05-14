from __future__ import print_function, division, absolute_import, unicode_literals
from TensorflowCode.core import SRNN_structure as SRNN, util, GTLabelProvider as label_provider, \
    image_util_SCN as image_util, unet_SCN as unet
import numpy as np
import click
import os
import logging
from datetime import datetime

t = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" #按照顺序排列GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--saliency', default='Base_Unet', help='Key word for this running time')
@click.option('--run_times', default=1, type=click.IntRange(min=1, clamp=True), help='network training times')
@click.option('--time', default=t, help='the current time or the time when the model to restore was trained')
@click.option('--trainer_learning_rate', default=0.001, type=click.FloatRange(min=1e-8, clamp=True),
              help='network learning rate')
@click.option('--train_validation_batch_size', default=5, type=click.IntRange(min=1),
              help='the number of validation cases')
@click.option('--test_n_files', default=15, type=click.IntRange(min=1), help='the number of test cases')
@click.option('--train_original_search_path', default='../dataset/train_original/*.nii.gz',
              help='search pattern to find all original training data and label images')
@click.option('--srnn_search_path', default='../dataset/crop/*.png',
              help='search pattern to find all ground truth label to train SRNN')
@click.option('--train_search_path', default='../dataset/train_data_2d/*.png',
              help='search pattern to find all training data and label images')
@click.option('--train_data_suffix', default='_img.png', help='suffix pattern for the training data images')
@click.option('--train_label_suffix', default='_lab.png', help='suffix pattern for the training label images')
@click.option('--train_shuffle_data', default=True, type=bool,
              help='whether the order of training files should be randomized after each epoch')
@click.option('--train_crop_patch', default=False, type=bool,
              help='whether patches of a certain size need to be cropped for training')
@click.option('--train_patch_size', default=(-1, -1, -1),
              type=(click.IntRange(min=-1), click.IntRange(min=-1), click.IntRange(min=-1)),
              help='size of the training patches')
@click.option('--train_channels', default=1, type=click.IntRange(min=1), help='number of training data channels')
@click.option('--train_n_class', default=4, type=click.IntRange(min=1),
              help='number of training label classes, including the background')
@click.option('--train_contain_foreground', default=False, type=bool,
              help='if the training patches should contain foreground')
@click.option('--train_label_intensity', default=(0, 88, 200, 244), multiple=True,
              type=click.IntRange(min=0), help='list of intensities of the training ground truths')
@click.option('--net_layers', default=5, type=click.IntRange(min=2),
              help='number of convolutional blocks in the down-sampling path')
@click.option('--net_features_root', default=16, type=click.IntRange(min=1),
              help='number of features of the first convolution layer')
@click.option('--net_cost_name', default=u'exponential_logarithmic',
              type=click.Choice(["cross_entropy", "weighted_cross_entropy", "dice_loss",
                                 "generalized_dice_loss", "cross_entropy+dice_loss",
                                 "weighted_cross_entropy+generalized_dice_loss",
                                 "exponential_logarithmic"]), help='type of the cost function')
@click.option('--net_regularizer_type', default=None,
              type=click.Choice(['L2_norm', 'L1_norm', 'anatomical_constraint']),
              help='type of regularization')
@click.option('--net_regularization_coefficient', default=5e-4, type=click.FloatRange(min=0),
              help='regularization coefficient')
@click.option('--net_srnn_model_path', default='./autoencoder_trained_%s',
              help='path where to restore the SRNN auto-encoder parameters for regularization')
@click.option('--trainer_batch_size', default=32, type=click.IntRange(min=1, clamp=True),
              help='batch size for each training iteration')
@click.option('--trainer_optimizer_name', default='adam', type=click.Choice(['momentum', 'adam']),
              help='type of the optimizer to use (momentum or adam)')
@click.option('--train_model_path', default='./unet_trained_%s_%s/No_%d', help='path where to store checkpoints')
@click.option('--train_training_iters', default=638, type=click.IntRange(min=1),
              help='number of training iterations during each epoch')
@click.option('--train_epochs', default=30, type=click.IntRange(min=1), help='number of epochs')
@click.option('--train_dropout_rate', default=0.2, type=click.FloatRange(min=0, max=1), help='dropout probability')
@click.option('--train_clip_gradient', default=False, type=bool,
              help='whether to apply gradient clipping with L2 norm threshold 1.0')
@click.option('--train_display_step', default=100, type=click.IntRange(min=1),
              help='number of steps till outputting stats')
@click.option('--train_prediction_path', default='./validation_prediction_%s_%s/No_%d',
              help='path where to save predictions on each epoch')
@click.option('--train_restore', default=False, type=bool, help='whether previous model checkpoint need restoring')
@click.option('--test_search_path', default='../dataset/test_data/*.nii.gz',
              help='a search pattern to find all test data and label images')
@click.option('--test_data_suffix', default='_img.nii.gz', help='suffix pattern for the test data images')
@click.option('--test_label_suffix', default='_lab.nii.gz', help='suffix pattern for the test label images')
@click.option('--test_shuffle_data', default=False, type=bool,
              help='whether the order of the loaded test files path should be randomized')
@click.option('--test_channels', default=1, type=click.IntRange(min=1), help='number of test data channels')
@click.option('--test_n_class', default=4, type=click.IntRange(min=1),
              help='number of test label classes, including the background')
@click.option('--test_label_intensity', default=(0, 88, 200, 244), multiple=True,
              type=click.IntRange(min=0),
              help='tuple of intensities of the test ground truths')
@click.option('--test_prediction_path', default=u'./test_prediction_%s_%s/No_%d',
              help='path where to save test predictions')
@click.option('--val_search_path', default='../dataset/val_data/*.nii.gz',
              help='a search pattern to find all validation data and label images')
@click.option('--val_data_suffix', default='_img.nii.gz', help='suffix pattern for the val data images')
@click.option('--val_label_suffix', default='_lab.nii.gz', help='suffix pattern for the val label images')
@click.option('--val_shuffle_data', default=False, type=bool,
              help='whether the order of the loaded val files path should be randomized')
@click.option('--val_channels', default=1, type=click.IntRange(min=1), help='number of val data channels')
@click.option('--val_n_class', default=4, type=click.IntRange(min=1),
              help='number of val label classes, including the background')
@click.option('--val_label_intensity', default=(0, 88, 200, 244), multiple=True,
              type=click.IntRange(min=0),
              help='tuple of intensities of the test ground truths')
@click.option('--train_center_crop', default=True, type=bool,
              help='whether to extract roi from center during training')
@click.option('--train_center_roi', default=(120, 120, 1), multiple=True, type=click.IntRange(min=0),
              help='roi size you want to extract during training')
@click.option('--test_center_crop', default=True, type=bool,
              help='whether to extract roi from center while testing')
@click.option('--test_center_roi', default=(120, 120, 1), multiple=True, type=click.IntRange(min=0),
              help='roi size you want to extract while testing')
@click.option('--scn_parameter', default=5e-4, type=click.FloatRange(min=0),
              help='weight of spatial constraint')
@click.option('--scn_button', default=False, type=bool,
              help='whether to use spatial constraint')
def run(run_times, time, train_search_path, train_data_suffix, train_label_suffix, train_shuffle_data, train_crop_patch,
        train_patch_size, train_channels, train_n_class, train_contain_foreground, train_label_intensity,
        train_original_search_path, net_layers, net_features_root, net_cost_name, net_regularizer_type,
        net_regularization_coefficient, net_srnn_model_path, trainer_batch_size, trainer_optimizer_name,
        trainer_learning_rate, train_validation_batch_size, train_model_path, train_training_iters, train_epochs,
        train_dropout_rate, train_clip_gradient, train_display_step, train_prediction_path, train_restore,
        test_search_path, test_data_suffix, test_label_suffix, test_shuffle_data, test_channels, test_n_class,
        test_label_intensity, test_n_files, test_prediction_path, val_search_path, val_data_suffix, val_label_suffix,
        val_shuffle_data, val_channels, val_n_class, val_label_intensity, saliency, train_center_crop, train_center_roi,
        test_center_crop, test_center_roi, scn_parameter, scn_button, srnn_search_path
        ):

    if train_restore:
        assert time != t, "The time when the model to restore was trained is not the time now! " #断言

    train_acc_table = np.array([])
    train_dice_table = np.array([])
    train_auc_table = np.array([])
    train_sens_table = np.array([])
    train_spec_table = np.array([])

    test_acc_table = np.array([])
    test_dice_table = np.array([])
    test_auc_table = np.array([])

    for i in range(run_times):
        train_data_provider = image_util.ImageDataProvider(search_path=train_search_path,
                                                           data_suffix=train_data_suffix,
                                                           label_suffix=train_label_suffix,
                                                           shuffle_data=train_shuffle_data,
                                                           crop_patch=train_crop_patch,
                                                           patch_size=train_patch_size,
                                                           channels=train_channels,
                                                           n_class=train_n_class,
                                                           contain_foreground=train_contain_foreground,
                                                           label_intensity=train_label_intensity,
                                                           center_crop=train_center_crop,
                                                           center_roi=train_center_roi,
                                                           inference_phase=False
                                                           )

        SRNN_data_provider = label_provider.GTLabelProvider(search_path=srnn_search_path,
                                                            label_suffix=train_label_suffix,
                                                            shuffle_data=train_shuffle_data,
                                                            channels=train_channels,
                                                            n_class=train_n_class,
                                                            label_intensity=train_label_intensity,
                                                            center_crop=True
                                                            )

        train_original_data_provider = image_util.ImageDataProvider(search_path=train_original_search_path,
                                                                    data_suffix=test_data_suffix,
                                                                    label_suffix=test_label_suffix,
                                                                    shuffle_data=False,
                                                                    crop_patch=False,
                                                                    patch_size=train_patch_size,
                                                                    channels=train_channels,
                                                                    n_class=train_n_class,
                                                                    contain_foreground=train_contain_foreground,
                                                                    label_intensity=train_label_intensity,
                                                                    center_crop=train_center_crop,
                                                                    center_roi=train_center_roi,
                                                                    inference_phase=True
                                                           )

        test_data_provider = image_util.ImageDataProvider(search_path=test_search_path,
                                                          data_suffix=test_data_suffix,
                                                          label_suffix=test_label_suffix,
                                                          shuffle_data=test_shuffle_data,
                                                          crop_patch=False,
                                                          channels=test_channels,
                                                          n_class=test_n_class,
                                                          label_intensity=test_label_intensity,
                                                          center_crop=test_center_crop,
                                                          center_roi=test_center_roi,
                                                          inference_phase=True)

        val_data_provider = image_util.ImageDataProvider(search_path=val_search_path,
                                                         data_suffix=val_data_suffix,
                                                         label_suffix=val_label_suffix,
                                                         shuffle_data=val_shuffle_data,
                                                         crop_patch=False,
                                                         channels=val_channels,
                                                         n_class=val_n_class,
                                                         label_intensity=val_label_intensity,
                                                         center_crop=test_center_crop,
                                                         center_roi=test_center_roi,
                                                         inference_phase=True)

        print("lalalala")
        if net_regularizer_type == 'anatomical_constraint':
            if os.path.exists(net_srnn_model_path % saliency):
                print("SRNN has already been trained.")
            else:
                logging.info("Train SRNN with 45 patient data...")
                srnn = SRNN.AutoEncoder(batch_size=trainer_batch_size,
                                        cost_kwargs={'regularizer_type': 'L1_norm',
                                                     'regularization_coefficient': 5e-4})
                srnn.train(train_data_provider, net_srnn_model_path % saliency)

                # logging.info("Train SRNN with additional ground truth labels...")
                # tf.reset_default_graph()
                # srnn_2 = SRNN.AutoEncoder(batch_size=trainer_batch_size,
                #                           cost_kwargs={'regularizer_type': 'L2_norm',
                #                                        'regularization_coefficient': 5e-4})
                # srnn_2.train(SRNN_data_provider, net_srnn_model_path % saliency, training_iters=34, restore=True)
                logging.info("Done pre-train.")

        net = unet.UNet(layers=net_layers, features_root=net_features_root, channels=train_channels,
                        n_class=train_n_class, batch_size=trainer_batch_size, cost_name=net_cost_name,
                        sc_coefficient=scn_parameter, need_sc=scn_button,
                        cost_kwargs={'regularizer_type': net_regularizer_type,
                                     'regularization_coefficient': net_regularization_coefficient,
                                     'srnn_model_path': (net_srnn_model_path % saliency)})

        trainer = unet.Trainer(net, batch_size=trainer_batch_size, optimizer_name=trainer_optimizer_name,
                               opt_kwargs={'learning_rate': trainer_learning_rate}, dropout=train_dropout_rate)

        path, train_acc, train_dice, train_auc, train_sens, train_spec = trainer.train(train_data_provider,
                                                                                       val_data_provider,
                                                                                       train_original_data_provider,
                                                                                       train_validation_batch_size,
                                                                                       model_path=train_model_path
                                                                                       % (saliency, time, i),
                                                                                       training_iters=train_training_iters,
                                                                                       epochs=train_epochs,
                                                                                       clip_gradient=train_clip_gradient,
                                                                                       display_step=train_display_step,
                                                                                       prediction_path=train_prediction_path
                                                                                       % (saliency, time, i),
                                                                                       restore=train_restore)
        train_acc_table = np.hstack((train_acc_table, train_acc))
        train_dice_table = np.hstack((train_dice_table, train_dice))
        train_auc_table = np.hstack((train_auc_table, train_auc))
        train_sens_table = np.hstack((train_sens_table, train_sens))
        train_spec_table = np.hstack((train_spec_table, train_spec))

        train_summary_path = './train_summary_%s_%s' % (saliency, time)
        if not os.path.exists(train_summary_path):
            logging.info('Allocating {:}'.format(train_summary_path))
            os.makedirs(train_summary_path)
        np.savez(os.path.join(train_summary_path, 'No_%d.npz' % i), acc=train_acc, dice=train_dice, auc=train_auc,
                 sens=train_sens, spec=train_spec)

        test_data_provider.reset_index()
        test_data, test_labels, test_affine, _ = test_data_provider(test_n_files)
        predictions = net.predict(path, test_data)

        test_acc = unet.acc_rate(predictions, test_labels)
        test_dice = unet.dice_score(predictions, test_labels)
        test_auc = unet.auc_score(predictions, test_labels)

        test_acc_table = np.hstack((test_acc_table, test_acc))
        test_dice_table = np.hstack((test_dice_table, test_dice))
        test_auc_table = np.hstack((test_auc_table, test_auc))

        dice_score_path = './dice_score_%s_%s' % (saliency, time)
        if not os.path.exists(dice_score_path):
            logging.info('Allocating {:}'.format(dice_score_path))
            os.makedirs(dice_score_path)
        np.save(os.path.join(dice_score_path, 'No_%d.npy' % i), test_dice)
        print("##################################################")
        print("Mean Dice score= {:.4f}".format(np.mean(test_dice)))

        for j in range(len(test_data)):
            test_data[j] = np.expand_dims(test_data[j], axis=0).transpose((0, 2, 3, 1, 4))
            test_labels[j] = np.expand_dims(test_labels[j], axis=0).transpose((0, 2, 3, 1, 4))
            predictions[j] = np.expand_dims(predictions[j], axis=0).transpose((0, 2, 3, 1, 4))

        util.save_prediction(test_data, test_labels, predictions, test_prediction_path % (saliency, time, i))
        util.save_prediction_1(predictions, test_affine, test_prediction_path % (saliency, time, i))
        util.save_prediction_2(predictions, test_prediction_path % (saliency, time, i))

        test_summary_path = './test_summary_%s_%s' % (saliency, time)
        if not os.path.exists(test_summary_path):
            logging.info('Allocating {:}'.format(test_summary_path))
            os.makedirs(test_summary_path)
        np.savez(os.path.join(test_summary_path, 'No_%d.npz' % i), acc=test_acc, dice=test_dice, auc=test_auc)

    mean_train_acc = np.mean(np.reshape(train_acc_table, [run_times, -1]), axis=0)
    mean_train_dice = np.mean(np.reshape(train_dice_table, [run_times, -1]), axis=0)
    mean_train_auc = np.mean(np.reshape(train_auc_table, [run_times, -1]), axis=0)
    mean_train_sens = np.mean(np.reshape(train_sens_table, [run_times, -1]), axis=0)
    mean_train_spec = np.mean(np.reshape(train_spec_table, [run_times, -1]), axis=0)
    mean_test_acc = np.mean(np.reshape(test_acc_table, [run_times, -1]), axis=0)
    mean_test_dice = np.mean(np.reshape(train_dice_table, [run_times, -1]), axis=0)
    mean_test_auc = np.mean(np.reshape(train_auc_table, [run_times, -1]), axis=0)

    np.savez('./mean_train_summary_%s_%s.npz' % (saliency, time), acc=mean_train_acc, auc=mean_train_auc,
             sens=mean_train_sens, spec=mean_train_spec, dice=mean_train_dice)
    np.savez('./mean_test_summary_%s_%s.npz' % (saliency, time), acc=mean_test_acc, auc=mean_test_auc,
             dice=mean_test_dice)


if __name__ == '__main__':
    run()
