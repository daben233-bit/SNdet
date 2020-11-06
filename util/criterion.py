import torch
import numpy as np
from torch.nn.modules.loss import *
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

def criterion(output: Tensor, groundtruth: list, batch_size: int, neg_pos_ratio: int) -> Tensor:
    '''
    build the loss funciton，
    including pixel loss 、 affinity loss 、 repulsive loss three parts
    attention:
        affinity loss  only calculate the pixel points which the 'pixel mask' values are positive
        repulsive loss only calculate the pixel points which the 'shrink pixel mask' values are positive

    calculate the affinity link loss:
        if n_pos > 0:
            1. calculate the cross entropy loss on the whole affinity link map ( affinity_link_loss )
            2. filter the loss map by gt_affinity_link_mask , gt_affinity_link_weight_mask and the logistic label
                alink_mask = gt_affinity_link_mask == label ( label = 0 or 1 )
                alink_weight = gt_affinity_link_weight_mask * alink_mask
                number_links = sum(alink_weight)
                label_loss = sum( affinity_link_loss * alink_weight ) / number_links
            3. neg_loss = 2(label = 0)
                pos_loss = 2(label = 1)
            4. alink_loss = pos_loss + neg_loss * lambda
        else:
            pixel_neg_link_loss, pixel_pos_link_loss = .0, .0

    calculate the repulsive link loss:


    :param output: the output through neural network [ pixel_mask, affinity_link_mask, repulsive_link_mask ]
    :param groundtruth:generated groundtruth based on the given 4 points, also including the weight mask
                        [ gt_pixel_mask, gt_weight_mask,
                        gt_affinity_link, gt_affinity_link_weight_mask,
                        gt_repulsive_link_mask, gt_repulsive_link_weight_mask ]
    :return: final loss , the reduce sum result of three kinds of losses, is a Tensor with a single scalar
    '''
    pixel_mask, affinity_link_mask, repulsive_link_mask = output # [item.double() for item in output]

    # pixel_mask = torch.sigmoid(pixel_mask)
    # affinity_link_mask = torch.sigmoid(affinity_link_mask)
    # repulsive_link_mask = torch.sigmoid(repulsive_link_mask)

    gt_pixel_mask, gt_weight_mask, \
    gt_affinity_link, gt_affinity_link_weight_mask, \
    gt_repulsive_link, gt_repulsive_link_weight_mask, gt_repulsive_mask = groundtruth

    # calculate pixel loss
    selected_neg_mask, n_pos, n_neg = _OHNM(pixel_mask=pixel_mask, gt_pixel_mask=gt_pixel_mask,
                                            batch_size=batch_size, neg_pos_ratio=neg_pos_ratio)
    pixel_loss = nn.BCELoss(reduction='none')(pixel_mask, gt_pixel_mask.float())
    # print(pixel_loss)
    # print(pixel_loss.size())
    
    pixel_loss_weight_mask = gt_weight_mask + selected_neg_mask
    ploss = torch.sum(pixel_loss * pixel_loss_weight_mask) / (n_pos + n_neg)
    
    # calculate affinity link loss
    if n_pos > 0:
        affinity_link_loss = nn.BCELoss(reduction='none')(affinity_link_mask, gt_affinity_link.float())
        def get_aloss_by_label(label):
            alink_mask = gt_affinity_link == label
            alink_weight = gt_affinity_link_weight_mask * alink_mask
            number_alinks = torch.sum(alink_weight > 0)
            label_aloss = torch.sum(affinity_link_loss * alink_weight) / number_alinks
            return label_aloss

        alink_pos_loss, alink_neg_loss = get_aloss_by_label(1), get_aloss_by_label(0)
    else:
        alink_pos_loss, alink_neg_loss = .0, .0
    alink_loss = alink_pos_loss + alink_neg_loss  # * alink_lambda

    # calculate the repulsive link loss
    if n_pos > 0:
        repulsive_link_loss = nn.BCELoss(reduction='none')(repulsive_link_mask, gt_repulsive_link.float())
        def get_rloss_by_label(label):
            rlink_mask = (gt_repulsive_link == label) * gt_repulsive_mask
            if label == 1:
                rlink_weight = gt_repulsive_link_weight_mask * rlink_mask
                number_rlinks = torch.sum(rlink_weight > 0)
            else:
                rlink_weight = rlink_mask
                number_rlinks = torch.sum(rlink_mask > 0)
            # print(number_rlinks)
            label_rloss = torch.sum(repulsive_link_loss * rlink_weight) / number_rlinks
            # print(label_rloss)
            return label_rloss

        rlink_pos_loss = get_rloss_by_label(1)
        rlink_neg_loss = get_rloss_by_label(0)

    else:
        rlink_pos_loss, rlink_neg_loss = .0, .0
    rlink_loss = rlink_pos_loss + rlink_neg_loss  # * rlink_lambda
    print('ploss: %f, alinkposloss: %f, alinknegloss: %f, rlinkposloss: %f, rlinknegloss: %f' % (ploss, alink_pos_loss, alink_neg_loss, rlink_pos_loss, rlink_neg_loss))
    final_loss = alink_loss + rlink_loss + ploss * 2  # * pixel_loss_lambda

    return final_loss

def _OHNM(pixel_mask: Tensor, gt_pixel_mask: Tensor, batch_size: int, neg_pos_ratio: int):
    '''
    0. initialize the selected_neg_mask = []
    1. calculate the selected_negative_pixel_mask for single image in batch_size
        1. get negative_score_output_mask, gt_negative_pixel_mask, gt_pixel_mask
        2. calculate the number of positive pixel in gt_pixel_mask
        3. number_negative_pixel = number_positive_pixel * neg_pos_ratio if number_positive_pixel > 0 else 10000
        4. calculate the number of positive pixel in gt_negative_pixel_mask (number_negative_pixel_entries)
        5. number_negative_pixel = number_negative_pixel_entries if number_negative_pixel > number_negative_pixel_entries
        6. if number_negative_pixel > 0:
                1. filter the negative_score_output_mask by gt_negative_pixel_mask
                2. get the top k min value of the filtered negative_score_output_mask
                3. choose the k-th min value being the threshold
                4. selected_negative_pixel_mask_for_single_image =
                        logical_and( gt_negative_pixel_mask, filtered negative_score_output_mask <= threshold )
            else:
                selected_negative_pixel_mask_for_single_image = zero_like(negative_pixel_mask)
        7. append the selected_negative_pixel_mask_for_single_image to selected_neg_mask list
    2. stack the selected_neg_mask list and return it
    :param pixel_mask: output through the network of pixel confidence
    :param gt_pixel_mask: the groundtruth of pixel mask
    :return: selected_negative_pixel_mask for a batch
    '''
    selected_neg_mask = []
    n_pos = 0
    n_neg = 0
    pixel_mask_copy = pixel_mask.clone()
    gt_pixel_mask_copy = gt_pixel_mask.clone()
    for i in range(batch_size):
        negative_score_output_mask = pixel_mask_copy[i]
        gt_positive_pixel_mask = gt_pixel_mask_copy[i]
        gt_negative_pixel_mask = torch.logical_not(gt_positive_pixel_mask)
        number_positive_pixel = torch.sum(gt_positive_pixel_mask)
        # print(number_positive_pixel)
        number_negative_pixel = number_positive_pixel * neg_pos_ratio 
        # number_negative_pixel_entries = torch.sum(gt_negative_pixel_mask)
        # if number_negative_pixel > number_negative_pixel_entries:
        #     number_negative_pixel = number_negative_pixel_entries
        # if number_negative_pixel > 0:
        #     # print(number_negative_pixel)
        #     negative_score_output_mask *= gt_negative_pixel_mask
        #     # print(negative_score_output_mask)
        #     topk, _ = torch.topk(negative_score_output_mask.flatten(start_dim=0, end_dim=-1), k=int(number_negative_pixel), dim=-1, largest=True, sorted=True)
        #     threshold = topk[-1]
        #     # print(threshold)
        #     # print(gt_negative_pixel_mask.size())
        #     # print(negative_score_output_mask.size())
        #     selected_neg_mask_single = torch.logical_and(gt_negative_pixel_mask, (negative_score_output_mask >= threshold))
        # else:
        #     selected_neg_mask_single = torch.zeros_like(gt_negative_pixel_mask)
        # print(number_negative_pixel)
        negative_score_output_mask *= gt_negative_pixel_mask
        # print(negative_score_output_mask)
        topk, _ = torch.topk(negative_score_output_mask.flatten(start_dim=0, end_dim=-1), k=int(number_negative_pixel), dim=-1, largest=True, sorted=True)
        threshold = topk[-1]
        # print(threshold)
        # print(gt_negative_pixel_mask.size())
        # print(negative_score_output_mask.size())
        selected_neg_mask_single = torch.logical_and(gt_negative_pixel_mask, (negative_score_output_mask >= threshold))
        selected_neg_mask.append(selected_neg_mask_single)
        n_pos += number_positive_pixel
        n_neg += number_negative_pixel
    selected_neg_mask = torch.stack(selected_neg_mask)
    selected_neg_mask = selected_neg_mask * torch.true_divide(n_pos, n_pos + n_neg)
    return selected_neg_mask, n_pos, n_neg



if __name__ == '__main__':
    x = Tensor(
        [
            [0, 0.5, 0, 0, 0.6],
            [0, 0.5, 1, 1, 0],
            [0, 0, 1, 1, 0],
            [0, 0.5, 0, 0, 0]
        ]
    ).unsqueeze(0).unsqueeze(0)
    y = Tensor(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0]
        ]
    ).unsqueeze(0).unsqueeze(0)
    # print(x.size())
    selected_neg_mask = _OHNM(pixel_mask=x, gt_pixel_mask=y, batch_size=1, neg_pos_ratio=1)
    print(selected_neg_mask)
