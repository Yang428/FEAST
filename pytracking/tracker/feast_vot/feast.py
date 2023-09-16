from pytracking.tracker.base import BaseTracker
import torch
import torch.nn.functional as F
import torch.nn
import math
import time
import numpy as np
import cv2
import copy
from pytracking import dcf, fourier, TensorList, operation
from pytracking.features.preprocessing import numpy_to_torch
from pytracking.utils.plotting import show_tensor
from pytracking.libs.optimization import GaussNewtonCG, ConjugateGradient, GradientDescentL2
from .optim import ConvProblem, FactorizedConvProblem
from pytracking.features import augmentation, augmentation1
import ltr.data.processing_utils1 as prutils
from ltr import load_network

from pytracking.features.preprocessing import sample_patch_multiscale, sample_patch_transformed
import ltr.data.bounding_box_utils as bbutils
from pytracking.utils.plotting import show_tensor, plot_graph

from pytracking.bbox_fit import fit_bbox_to_mask
from pytracking.mask_to_disk import save_mask
import matplotlib.pyplot as plt
import pdb

class FEAST(BaseTracker):

    multiobj_mode = 'parallel'

    def initialize_features(self):
        if not getattr(self, 'features_initialized', False):
            self.params.net.initialize()
            self.params.features_filter.initialize()
        self.features_initialized = True

    def initialize(self, image, info: dict) -> dict:

        # Initialize some stuff
        self.frame_num = 1
        self.segm_update_num = 0           ## background updating frame number
        self.segm_update_flag = False      ## background updating flag

        if not hasattr(self.params, 'device'):
            self.params.device = 'cuda' if self.params.use_gpu else 'cpu'

        # Initialize features
        self.initialize_features()

        # The IouNet network
        self.net = self.params.net

        # Check if image is color
        self.params.features_filter.set_is_color(image.shape[2] == 3)

        # Get feature specific params
        self.fparams = self.params.features_filter.get_fparams('feature_params')

        self.time = 0
        ## Rotated bounding box in VOT benchmarks 
        state = info['init_bbox']
        self.gt_poly = np.array(state)
        x_ = np.array(state[::2])
        y_ = np.array(state[1::2])
        self.pos = torch.Tensor([np.mean(y_), np.mean(x_)])
        # overwrite state - needed for segmentation
        if self.params.vot_anno_conversion_type == 'preserve_area':
            state = self.poly_to_aabbox(x_, y_)
        else:
            state = np.array([np.min(x_), np.min(y_), np.max(x_) - np.min(x_), np.max(y_) - np.min(y_)])

        self.target_sz = torch.Tensor([state[3], state[2]])

        # Set search area
        self.target_scale = 1.0
        search_area = torch.prod(self.target_sz * self.params.search_area_scale).item()
        if search_area > self.params.max_image_sample_size:
            self.target_scale = math.sqrt(search_area / self.params.max_image_sample_size)
        elif search_area < self.params.min_image_sample_size:
            self.target_scale = math.sqrt(search_area / self.params.min_image_sample_size)

        # Target size in base scale
        self.base_target_sz = self.target_sz / self.target_scale
        self.pre_target_sz = self.target_sz.clone()
        # Use odd square search area and set sizes
        feat_max_stride = max(self.params.features_filter.stride())
        if getattr(self.params, 'search_area_shape', 'square') == 'square':
            self.img_sample_sz = torch.round(
                torch.sqrt(torch.prod(self.base_target_sz * self.params.search_area_scale))) * torch.ones(2)
        elif self.params.search_area_shape == 'initrect':
            self.img_sample_sz = torch.round(self.base_target_sz * self.params.search_area_scale)
        else:
            raise ValueError('Unknown search area shape')
        if self.params.feature_size_odd:
            self.img_sample_sz += feat_max_stride - self.img_sample_sz % (2 * feat_max_stride)
        else:
            self.img_sample_sz += feat_max_stride - (self.img_sample_sz + feat_max_stride) % (2 * feat_max_stride)

        # Set sizes
        self.img_support_sz = self.img_sample_sz
        self.feature_sz = self.params.features_filter.size(self.img_sample_sz)
        self.output_sz = self.params.score_upsample_factor * self.img_support_sz  # Interpolated size of the output
        self.kernel_size = self.fparams.attribute('kernel_size')

        # Optimization options
        self.params.precond_learning_rate = self.fparams.attribute('learning_rate')
        if self.params.CG_forgetting_rate is None or max(self.params.precond_learning_rate) >= 1:
            self.params.direction_forget_factor = 0
        else:
            self.params.direction_forget_factor = (1 - max(
                self.params.precond_learning_rate)) ** self.params.CG_forgetting_rate

        self.output_window = None
        if getattr(self.params, 'window_output', False):
            if getattr(self.params, 'use_clipped_window', False):
                self.output_window = dcf.hann2d_clipped(self.output_sz.long(),
                                                        self.output_sz.long() * self.params.effective_search_area / self.params.search_area_scale,
                                                        centered=False).to(self.params.device)
            else:
                self.output_window = dcf.hann2d(self.output_sz.long(), centered=False).to(self.params.device)

        # Initialize some learning things
        self.init_learning()

        # Convert image
        im = numpy_to_torch(image)
        self.im = im  # For debugging only

        # Setup scale bounds
        self.image_sz = torch.Tensor([im.shape[2], im.shape[3]])
        self.min_scale_factor = torch.max(10 / self.base_target_sz)
        self.max_scale_factor = torch.min(self.image_sz / self.base_target_sz)

        #### IouNet sizes ####
        self.feature_sz1 = torch.tensor([22., 22.])
        self.kernel_size1 = torch.tensor([4., 4.])
        sz = self.params.image_sample_size
        sz = torch.Tensor([sz, sz] if isinstance(sz, int) else sz)
        if self.params.get('use_image_aspect_ratio', False):
            sz = self.image_sz * sz.prod().sqrt() / self.image_sz.prod().sqrt()
            stride = self.params.get('feature_stride', 32)
            sz = torch.round(sz / stride) * stride
        self.target_sz1 = self.target_sz.clone()
        self.img_sample_sz1 = sz
        self.img_support_sz1 = self.img_sample_sz1
        search_area = torch.prod(self.target_sz1 * self.params.search_area_scale1).item()
        self.target_scale1 =  math.sqrt(search_area) / self.img_sample_sz1.prod().sqrt()
        self.base_target_sz1 = self.target_sz1 / self.target_scale1
        ######################
        # Extract and transform sample for IoUNet
        init_backbone_feat = self.generate_init_samples_IouNet(im)

        # Initialize IoUNet
        self.init_iou_net(init_backbone_feat)

        # Extract and transform sample for DCF
        x = self.generate_init_samples(im)

        # Initialize projection matrix
        self.init_projection_matrix(x)

        # Transform to get the training sample
        train_x = self.preprocess_sample(x)
  
        # Init memory
        self.init_memory(train_x)

        # Generate label function
        init_y = self.init_label_function(train_x)

        # Init optimizer and do initial optimization
        self.init_optimization(train_x, init_y)
      
        if self.params.use_segmentation:
            self.init_segmentation(image, state, init_mask=None)
 
        # array of scores
        self.scores = np.array([1])
        self.confidences = np.array([])
        self.confidence = 0.0
        self.vf_score = np.array([])

    def init_optimization(self, train_x, init_y):
        # Initialize filter
        filter_init_method = getattr(self.params, 'filter_init_method', 'zeros')
        self.filter = TensorList(
            [x.new_zeros(1, cdim, sz[0], sz[1]) for x, cdim, sz in zip(train_x, self.compressed_dim, self.kernel_size)])
        if filter_init_method == 'zeros':
            pass
        elif filter_init_method == 'randn':
            for f in self.filter:
                f.normal_(0, 1 / f.numel())
        else:
            raise ValueError('Unknown "filter_init_method"')

        # Get parameters
        self.params.update_projection_matrix = getattr(self.params, 'update_projection_matrix',
                                                       True) and self.params.use_projection_matrix
        optimizer = getattr(self.params, 'optimizer', 'GaussNewtonCG')

        # Setup factorized joint optimization
        if self.params.update_projection_matrix:
            self.joint_problem = FactorizedConvProblem(self.init_training_samples, init_y, self.filter_reg,
                                                       self.fparams.attribute('projection_reg'), self.params,
                                                       self.init_sample_weights,
                                                       self.projection_activation, self.response_activation)

            # Variable containing both filter and projection matrix
            joint_var = self.filter.concat(self.projection_matrix)

            # Initialize optimizer
            analyze_convergence = getattr(self.params, 'analyze_convergence', False)
            if optimizer == 'GaussNewtonCG':
                self.joint_optimizer = GaussNewtonCG(self.joint_problem, joint_var, plotting=(self.params.debug >= 3),
                                                     analyze=analyze_convergence, fig_num=(12, 13, 14))
            elif optimizer == 'GradientDescentL2':
                self.joint_optimizer = GradientDescentL2(self.joint_problem, joint_var,
                                                         self.params.optimizer_step_length,
                                                         self.params.optimizer_momentum,
                                                         plotting=(self.params.debug >= 3), debug=analyze_convergence,
                                                         fig_num=(12, 13))

            # Do joint optimization
            if isinstance(self.params.init_CG_iter, (list, tuple)):
                self.joint_optimizer.run(self.params.init_CG_iter)
            else:
                self.joint_optimizer.run(self.params.init_CG_iter // self.params.init_GN_iter, self.params.init_GN_iter)

            if analyze_convergence:
                opt_name = 'CG' if getattr(self.params, 'CG_optimizer', True) else 'GD'
                for val_name, values in zip(['loss', 'gradient'],
                                            [self.joint_optimizer.losses, self.joint_optimizer.gradient_mags]):
                    val_str = ' '.join(['{:.8e}'.format(v.item()) for v in values])
                    file_name = '{}_{}.txt'.format(opt_name, val_name)
                    with open(file_name, 'a') as f:
                        f.write(val_str + '\n')
                raise RuntimeError('Exiting')

        # Re-project samples with the new projection matrix
        compressed_samples = self.project_sample(self.init_training_samples, self.projection_matrix)
        for train_samp, init_samp in zip(self.training_samples, compressed_samples):
            train_samp[:init_samp.shape[0], ...] = init_samp

        self.hinge_mask = None

        # Initialize optimizer
        self.conv_problem = ConvProblem(self.training_samples, self.y, self.filter_reg, self.sample_weights,
                                        self.response_activation, self.wgt)

        if optimizer == 'GaussNewtonCG':
            self.filter_optimizer = ConjugateGradient(self.conv_problem, self.filter,
                                                      fletcher_reeves=self.params.fletcher_reeves,
                                                      direction_forget_factor=self.params.direction_forget_factor,
                                                      debug=(self.params.debug >= 3), fig_num=(12, 13))
        elif optimizer == 'GradientDescentL2':
            self.filter_optimizer = GradientDescentL2(self.conv_problem, self.filter, self.params.optimizer_step_length,
                                                      self.params.optimizer_momentum, debug=(self.params.debug >= 3),
                                                      fig_num=12)

        # Transfer losses from previous optimization
        if self.params.update_projection_matrix:
            self.filter_optimizer.residuals = self.joint_optimizer.residuals
            self.filter_optimizer.losses = self.joint_optimizer.losses

        if not self.params.update_projection_matrix:
            self.filter_optimizer.run(self.params.init_CG_iter)

        # Post optimization
        self.filter_optimizer.run(self.params.post_init_CG_iter)

        # Free memory
        del self.init_training_samples
        if self.params.use_projection_matrix:
            del self.joint_problem, self.joint_optimizer

    ## update segmentation samples ##
    def update_segm_sample(self, image, pos, sz):
        # pos and sz are in the image coordinates
        tlx_ = pos[1] - sz[1] / 2
        tly_ = pos[0] - sz[0] / 2
        w_ = sz[1]
        h_ = sz[0]
        bb = [tlx_.item(), tly_.item(), w_.item(), h_.item()]

        # extract patch
        patch, f_ = prutils.sample_target(image, np.array(bb), self.params.segm_surroundings_area_factor,
                                          output_sz=self.params.segm_output_sz)

        # normalize input image
        init_patch_norm_ = patch.astype(np.float32) / float(255)
        init_patch_norm_ -= self.params.segm_normalize_mean
        init_patch_norm_ /= self.params.segm_normalize_std

        # put image patch and mask to GPU
        patch_gpu = torch.Tensor(init_patch_norm_)
        if self.params.use_gpu:
            patch_gpu = patch_gpu.to(self.params.device)

            # reshape image for the feature extractor
            patch_gpu = torch.unsqueeze(patch_gpu, dim=0).permute(0, 3, 1, 2)

        # extract features (extracting twice on the same patch - not necessary)
        test_feat = self.segm_net.extract_backbone_features(patch_gpu)

        # prepare features in the list (format for the network)
        test_feat_segm = [feat for feat in test_feat.values()]

        train_masks = [self.init_mask_patch]
        if self.params.segm_use_dist:
            if self.params.segm_dist_map_type == 'center':
                # center-based distance map
                test_dist_map = [self.dist_map]
            elif self.params.segm_dist_map_type == 'bbox':
                # bbox-based distance map
                D = self.create_dist_gauss(self.params.segm_output_sz, w_.item() * f_, h_.item() * f_)
                test_dist_map = [torch.unsqueeze(torch.unsqueeze(torch.Tensor(D).to(self.params.device), dim=0), dim=0)]
        else:
            test_dist_map = None

        # Obtain segmentation prediction
        segm_pred = self.segm_net.segm_predictor(test_feat_segm, self.train_feat_segm, train_masks, test_dist_map, self.segm_feat_ups, self.up_masks, segm_update_flag=self.segm_update_flag)
        mask = F.softmax(segm_pred, dim=1)[0, 0, :, :].cpu().numpy()
        mask = (mask > self.params.segm_mask_thr).astype(np.uint8)

        # find the max area mask
        if cv2.__version__[-5] == '4':
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        else:
            _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt_area = [cv2.contourArea(cnt) for cnt in contours]
 
        if len(cnt_area) > 0 and len(contours) != 0 and np.max(cnt_area) > 50:
            contour = contours[np.argmax(cnt_area)]  # use max area polygon
            mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 1, thickness=-1)
            #pdb.set_trace()
            train_masks = [torch.unsqueeze(torch.unsqueeze(torch.tensor(mask.astype(np.float32)), dim=0), dim=0).to(self.params.device)]
            segm_vf = self.segm_net.segm_predictor(self.train_feat_segm, test_feat_segm, train_masks, [self.dist_map])
            mask_vf = (F.softmax(segm_vf, dim=1)[:,:1,...] > self.params.segm_mask_thr).float()
            vf_score = ((2*(mask_vf*self.init_mask_patch).sum()+1)/(mask_vf.sum()+self.init_mask_patch.sum()+1)).cpu().numpy()
            
            mask_gpu = train_masks[0]
            #mask_gpu = torch.unsqueeze(torch.unsqueeze(torch.tensor(mask).type(torch.float32), dim=0), dim=0).to(self.params.device)         
            if len(self.up_masks) == 3:   # size of memory bank is set to 4, if the number of samples is exceeded, pop the earliest one
                idx = self.vf_score.argmin()
                self.segm_feat_ups.pop(idx)
                self.up_masks.pop(idx)  
                self.vf_score = np.delete(self.vf_score, idx)      
            self.segm_feat_ups.append(test_feat_segm) # append new segmentation sample
            self.up_masks.append([mask_gpu]) 
            self.vf_score = np.append(self.vf_score, vf_score) 
            self.segm_update_flag = True  

    def track(self, image, info: dict = None) -> dict:

        self.frame_num += 1
        self.frame_name = '%08d' % self.frame_num

        # Convert image
        im = numpy_to_torch(image)
        self.im = im  # For debugging only

        # ------- LOCALIZATION ------- #

        # Get sample
        sample_pos = copy.deepcopy(self.pos)
        sample_scales = self.target_scale * self.params.scale_factors 
        test_x = self.extract_processed_sample(im, sample_pos, sample_scales, self.img_sample_sz)

        # Compute scores
        scores_raw = self.apply_filter(test_x)
        translation_vec, scale_ind, s, flag = self.localize_target(scores_raw)
        new_pos = sample_pos + translation_vec

        # Localization uncertainty and confidence threshold
        max_score = torch.max(s).item()
        uncert_score = 0
        if self.frame_num > 5:
            uncert_score = np.mean(self.scores) / max_score
        self.uncert_score = uncert_score
    
        if uncert_score < self.params.tracking_uncertainty_thr:
            self.scores = np.append(self.scores, max_score)
            if self.scores.size > self.params.response_budget_sz:
                self.scores = np.delete(self.scores, 0)

        if flag == 'not_found':
            uncert_score = 100
 
        # Update position and scale
        if uncert_score < self.params.tracking_uncertainty_thr:
            if getattr(self.params, 'use_classifier', True):
                self.update_state(new_pos, sample_scales[scale_ind])

        if self.params.debug >= 2:
            show_tensor(s[scale_ind, ...], 5, title='Max score = {:.2f}'.format(torch.max(s[scale_ind, ...]).item()))

        # just a sanity check so that it does not get out of image
        if new_pos[0] < 0:
            new_pos[0] = 0
        if new_pos[1] < 0:
            new_pos[1] = 0
        if new_pos[0] >= image.shape[0]:
            new_pos[0] = image.shape[0] - 1
        if new_pos[1] >= image.shape[1]:
            new_pos[1] = image.shape[1] - 1

        pred_region_xywh = None
        if self.segmentation_task or (
            self.params.use_segmentation and uncert_score < self.params.uncertainty_segment_thr):
            pred_region_xywh, pred_region_rotate = self.segment_target(image, new_pos, self.target_sz)
            if pred_region_xywh is None:
                self.pos = new_pos.clone()
        else:
            self.pos = new_pos.clone()
        # ------- UPDATE ------- #

        # Check flags and set learning rate if hard negative
        update_flag = flag not in ['not_found', 'uncertain']
        hard_negative = (flag == 'hard_negative')
        learning_rate = self.params.hard_negative_learning_rate if hard_negative else None
        if uncert_score < self.params.tracking_uncertainty_thr:
            train_x = TensorList([x[scale_ind:scale_ind + 1, ...] for x in test_x]) # Get train sample
            train_y = self.get_label_function(sample_pos, sample_scales[scale_ind])   # Create label for sample
            self.update_memory(train_x, train_y, learning_rate)                     # Update memory for DCF
            ## update segmentation sample
            #'''
            if (hard_negative or self.segm_hard_neg) and (self.frame_num-self.segm_update_num)>(self.params.segm_update_skip-1):
                # if there is hard negative in background, update 10 skipping
                self.update_segm_sample(image, new_pos, self.target_sz)
                self.segm_update_num = self.frame_num
            #''' 

        # Train filter
        if hard_negative:
            self.filter_optimizer.run(self.params.hard_negative_CG_iter)
        elif (self.frame_num - 1) % self.params.train_skipping == 0:
            self.filter_optimizer.run(self.params.CG_iter)
 
        # ------- BOUNDING BOX REFINE ------- #    
        new_state = torch.cat((self.pos[[1,0]] - (self.target_sz[[1,0]]-1)/2, self.target_sz[[1,0]])) 
        if pred_region_xywh is not None: # when the predicted mask is None, using the module
            self.target_sz1 = self.pre_target_sz
            self.target_scale1 =  math.sqrt(torch.prod(self.target_sz1 * self.params.search_area_scale1).item()) / self.img_sample_sz1.prod().sqrt()
            self.pre_target_sz[0],self.pre_target_sz[1] = pred_region_xywh[3],pred_region_xywh[2]
            out = {'target_bbox': pred_region_rotate}
        else:
            # Extract backbone features
            backbone_feat, sample_coords, im_patches = self.extract_backbone_features(im, self.get_centered_sample_pos(),
                                                                          self.target_scale1 * self.params.scale_factors,
                                                                          self.img_sample_sz1)
            # Location of sample
            sample_pos1, sample_scales1 = self.get_sample_location(sample_coords)

            # Refine bounding box
            update_scale_flag = self.params.get('update_scale_when_uncertain', True) or flag != 'uncertain'
            self.refine_target_box(backbone_feat, sample_pos1[scale_ind[0],:], sample_scales1[scale_ind[0]], scale_ind[0], update_scale_flag)
            if hasattr(self, 'pos_iounet'):
                new_state = torch.cat((self.pos_iounet[[1,0]] - (self.target_sz_iounet[[1,0]]-1)/2, self.target_sz_iounet[[1,0]])) 
                self.pos = self.pos_iounet
            out = {'target_bbox': new_state.tolist()}

        return out


    def apply_filter(self, sample_x: TensorList):
        return operation.conv2d(sample_x, self.filter, mode='same')

    def localize_target(self, scores_raw):
        # Weighted sum (if multiple features) with interpolation in fourier domain
        weight = self.fparams.attribute('translation_weight', 1.0)
        scores_raw = weight * scores_raw
        sf_weighted = fourier.cfft2(scores_raw) / (scores_raw.size(2) * scores_raw.size(3))
        for i, (sz, ksz) in enumerate(zip(self.feature_sz, self.kernel_size)):
            sf_weighted[i] = fourier.shift_fs(sf_weighted[i],
                                              math.pi * (1 - torch.Tensor([ksz[0] % 2, ksz[1] % 2]) / sz))

        scores_fs = fourier.sum_fs(sf_weighted)
        scores = fourier.sample_fs(scores_fs, self.output_sz)

        if self.output_window is not None and not getattr(self.params, 'perform_hn_without_windowing', False):
            scores *= self.output_window

        if getattr(self.params, 'advanced_localization', False):
            return self.localize_advanced(scores)

        # Get maximum
        max_score, max_disp = dcf.max2d(scores)
        _, scale_ind = torch.max(max_score, dim=0)
        max_disp = max_disp.float().cpu()

        # Convert to displacements in the base scale
        disp = (max_disp + self.output_sz / 2) % self.output_sz - self.output_sz / 2

        # Compute translation vector and scale change factor
        translation_vec = disp[scale_ind, ...].view(-1) * (self.img_support_sz / self.output_sz) * self.target_scale
        translation_vec *= self.params.scale_factors[scale_ind]

        # Shift the score output for visualization purposes
        if self.params.debug >= 2:
            sz = scores.shape[-2:]
            scores = torch.cat([scores[..., sz[0] // 2:, :], scores[..., :sz[0] // 2, :]], -2)
            scores = torch.cat([scores[..., :, sz[1] // 2:], scores[..., :, :sz[1] // 2]], -1)

        return translation_vec, scale_ind, scores, None

    def localize_advanced(self, scores):
        """Does the advanced localization with hard negative detection and target not found."""

        sz = scores.shape[-2:]

        if self.output_window is not None and getattr(self.params, 'perform_hn_without_windowing', False):
            scores_orig = scores.clone()

            scores_orig = torch.cat([scores_orig[..., (sz[0] + 1) // 2:, :], scores_orig[..., :(sz[0] + 1) // 2, :]],
                                    -2)
            scores_orig = torch.cat([scores_orig[..., :, (sz[1] + 1) // 2:], scores_orig[..., :, :(sz[1] + 1) // 2]],
                                    -1)

            scores *= self.output_window

        # Shift scores back
        scores = torch.cat([scores[..., (sz[0] + 1) // 2:, :], scores[..., :(sz[0] + 1) // 2, :]], -2)
        scores = torch.cat([scores[..., :, (sz[1] + 1) // 2:], scores[..., :, :(sz[1] + 1) // 2]], -1)
        
        # Find maximum
        max_score1, max_disp1 = dcf.max2d(scores)
        _, scale_ind = torch.max(max_score1, dim=0)
        max_score1 = max_score1[scale_ind]
        max_disp1 = max_disp1[scale_ind, ...].float().cpu().view(-1)
        target_disp1 = max_disp1 - self.output_sz // 2
        translation_vec1 = target_disp1 * (self.img_support_sz / self.output_sz) * self.target_scale

        if max_score1.item() < self.params.target_not_found_threshold:
            return translation_vec1, scale_ind, scores, 'not_found'

        if self.output_window is not None and getattr(self.params, 'perform_hn_without_windowing', False):
            scores = scores_orig

        # Mask out target neighborhood
        target_neigh_sz = self.params.target_neighborhood_scale * self.target_sz / self.target_scale
        tneigh_top = max(round(max_disp1[0].item() - target_neigh_sz[0].item() / 2), 0)
        tneigh_bottom = min(round(max_disp1[0].item() + target_neigh_sz[0].item() / 2 + 1), sz[0])
        tneigh_left = max(round(max_disp1[1].item() - target_neigh_sz[1].item() / 2), 0)
        tneigh_right = min(round(max_disp1[1].item() + target_neigh_sz[1].item() / 2 + 1), sz[1])
        scores_masked = scores[scale_ind:scale_ind + 1, ...].clone()
        scores_masked[..., tneigh_top:tneigh_bottom, tneigh_left:tneigh_right] = 0

        # Find new maximum
        max_score2, max_disp2 = dcf.max2d(scores_masked)
        max_disp2 = max_disp2.float().cpu().view(-1)
        target_disp2 = max_disp2 - self.output_sz // 2
        translation_vec2 = target_disp2 * (self.img_support_sz / self.output_sz) * self.target_scale

        # Handle the different cases
        if max_score2 > self.params.distractor_threshold * max_score1:
            disp_norm1 = torch.sqrt(torch.sum(target_disp1 ** 2))
            disp_norm2 = torch.sqrt(torch.sum(target_disp2 ** 2))
            disp_threshold = self.params.dispalcement_scale * math.sqrt(sz[0] * sz[1]) / 2

            if disp_norm2 > disp_threshold and disp_norm1 < disp_threshold:
                return translation_vec1, scale_ind, scores, 'hard_negative'
            if disp_norm2 < disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec2, scale_ind, scores, 'hard_negative'
            if disp_norm2 > disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec1, scale_ind, scores, 'uncertain'

            # If also the distractor is close, return with highest score
            return translation_vec1, scale_ind, scores, 'uncertain'

        if max_score2 > self.params.hard_negative_threshold * max_score1 and max_score2 > self.params.target_not_found_threshold:
            return translation_vec1, scale_ind, scores, 'hard_negative'

        return translation_vec1, scale_ind, scores, None

    def extract_sample(self, im: torch.Tensor, pos: torch.Tensor, scales, sz: torch.Tensor):
        return self.params.features_filter.extract(im, pos, scales, sz)

    def extract_processed_sample(self, im: torch.Tensor, pos: torch.Tensor, scales, sz: torch.Tensor) -> (
    TensorList, TensorList):
        x = self.extract_sample(im, pos, scales, sz)
        return self.preprocess_sample(self.project_sample(x))

    def preprocess_sample(self, x: TensorList) -> (TensorList, TensorList):
        if getattr(self.params, '_feature_window', False):
            x = x * self.feature_window
        return x

    def project_sample(self, x: TensorList, proj_matrix=None):
        # Apply projection matrix
        if proj_matrix is None:
            proj_matrix = self.projection_matrix
        return operation.conv2d(x, proj_matrix).apply(self.projection_activation)

    def init_learning(self):
        # Get window function
        self.feature_window = TensorList([dcf.hann2d(sz).to(self.params.device) for sz in self.feature_sz])

        # Filter regularization
        self.filter_reg = self.fparams.attribute('filter_reg')

        # Activation function after the projection matrix (phi_1 in the paper)
        projection_activation = getattr(self.params, 'projection_activation', 'none')
        if isinstance(projection_activation, tuple):
            projection_activation, act_param = projection_activation

        if projection_activation == 'none':
            self.projection_activation = lambda x: x
        elif projection_activation == 'relu':
            self.projection_activation = torch.nn.ReLU(inplace=True)
        elif projection_activation == 'elu':
            self.projection_activation = torch.nn.ELU(inplace=True)
        elif projection_activation == 'mlu':
            self.projection_activation = lambda x: F.elu(F.leaky_relu(x, 1 / act_param), act_param)
        else:
            raise ValueError('Unknown activation')

        # Activation function after the output scores (phi_2 in the paper)
        response_activation = getattr(self.params, 'response_activation', 'none')
        if isinstance(response_activation, tuple):
            response_activation, act_param = response_activation

        if response_activation == 'none':
            self.response_activation = lambda x: x
        elif response_activation == 'relu':
            self.response_activation = torch.nn.ReLU(inplace=True)
        elif response_activation == 'elu':
            self.response_activation = torch.nn.ELU(inplace=True)
        elif response_activation == 'mlu':
            self.response_activation = lambda x: F.elu(F.leaky_relu(x, 1 / act_param), act_param)
        else:
            raise ValueError('Unknown activation')

    def generate_init_samples(self, im: torch.Tensor) -> TensorList:
        """Generate augmented initial samples."""

        # Compute augmentation size
        aug_expansion_factor = getattr(self.params, 'augmentation_expansion_factor', None)
        aug_expansion_sz = self.img_sample_sz.clone()
        aug_output_sz = None
        if aug_expansion_factor is not None and aug_expansion_factor != 1:
            aug_expansion_sz = (self.img_sample_sz * aug_expansion_factor).long()
            aug_expansion_sz += (aug_expansion_sz - self.img_sample_sz.long()) % 2
            aug_expansion_sz = aug_expansion_sz.float()
            aug_output_sz = self.img_sample_sz.long().tolist()

        # Random shift operator
        get_rand_shift = lambda: None
        random_shift_factor = getattr(self.params, 'random_shift_factor', 0)
        if random_shift_factor > 0:
            get_rand_shift = lambda: ((torch.rand(2) - 0.5) * self.img_sample_sz * random_shift_factor).long().tolist()

        # Create transofmations
        self.transforms = [augmentation.Identity(aug_output_sz)]
        if 'shift' in self.params.augmentation:
            self.transforms.extend(
                [augmentation.Translation(shift, aug_output_sz) for shift in self.params.augmentation['shift']])
        if 'relativeshift' in self.params.augmentation:
            get_absolute = lambda shift: (torch.Tensor(shift) * self.img_sample_sz / 2).long().tolist()
            self.transforms.extend([augmentation.Translation(get_absolute(shift), aug_output_sz) for shift in
                                    self.params.augmentation['relativeshift']])
        if 'fliplr' in self.params.augmentation and self.params.augmentation['fliplr']:
            self.transforms.append(augmentation.FlipHorizontal(aug_output_sz, get_rand_shift()))
        if 'blur' in self.params.augmentation:
            self.transforms.extend([augmentation.Blur(sigma, aug_output_sz, get_rand_shift()) for sigma in
                                    self.params.augmentation['blur']])
        if 'scale' in self.params.augmentation:
            self.transforms.extend(
                [augmentation.Scale(scale_factor, aug_output_sz, get_rand_shift()) for scale_factor in
                 self.params.augmentation['scale']])
        if 'rotate' in self.params.augmentation:
            self.transforms.extend([augmentation.Rotate(angle, aug_output_sz, get_rand_shift()) for angle in
                                    self.params.augmentation['rotate']])

        init_samples = self.params.features_filter.extract_transformed(im, self.pos.round(), self.target_scale,
                                                                       aug_expansion_sz, self.transforms)

        # Remove augmented samples for those that shall not have
        for i, use_aug in enumerate(self.fparams.attribute('use_augmentation')):
            if not use_aug:
                init_samples[i] = init_samples[i][0:1, ...]

        # Add dropout samples
        if 'dropout' in self.params.augmentation:
            num, prob = self.params.augmentation['dropout']
            self.transforms.extend(self.transforms[:1] * num)
            for i, use_aug in enumerate(self.fparams.attribute('use_augmentation')):
                if use_aug:
                    init_samples[i] = torch.cat([init_samples[i],
                                                 F.dropout2d(init_samples[i][0:1, ...].expand(num, -1, -1, -1), p=prob,
                                                             training=True)])

        return init_samples

    def init_projection_matrix(self, x):
        # Set if using projection matrix
        self.params.use_projection_matrix = getattr(self.params, 'use_projection_matrix', True)

        if self.params.use_projection_matrix:
            self.compressed_dim = self.fparams.attribute('compressed_dim', None)

            proj_init_method = getattr(self.params, 'proj_init_method', 'pca')
            if proj_init_method == 'pca':
                x_mat = TensorList([e.permute(1, 0, 2, 3).reshape(e.shape[1], -1).clone() for e in x])
                x_mat -= x_mat.mean(dim=1, keepdim=True)
                cov_x = x_mat @ x_mat.t()
                self.projection_matrix = TensorList(
                    [None if cdim is None else torch.svd(C)[0][:, :cdim].t().unsqueeze(-1).unsqueeze(-1).clone() for
                     C, cdim in
                     zip(cov_x, self.compressed_dim)])
            elif proj_init_method == 'randn':
                self.projection_matrix = TensorList(
                    [None if cdim is None else ex.new_zeros(cdim, ex.shape[1], 1, 1).normal_(0,
                                                                                             1 / math.sqrt(ex.shape[1]))
                     for ex, cdim in
                     zip(x, self.compressed_dim)])
        else:
            self.compressed_dim = x.size(1)
            self.projection_matrix = TensorList([None] * len(x))

    def init_label_function(self, train_x):
        # Allocate label function
        self.y = TensorList([x.new_zeros(self.params.sample_memory_size, 1, x.shape[2], x.shape[3]) for x in train_x])

        # Output sigma factor
        output_sigma_factor = self.fparams.attribute('output_sigma_factor')
        self.sigma = (
                     self.feature_sz / self.img_support_sz * self.base_target_sz).prod().sqrt() * output_sigma_factor * torch.ones(
            2)

        # Center pos in normalized coords
        target_center_norm = (self.pos - self.pos.round()) / (self.target_scale * self.img_support_sz)

        # Generate label functions
        for y, sig, sz, ksz, x in zip(self.y, self.sigma, self.feature_sz, self.kernel_size, train_x):
            center_pos = sz * target_center_norm + 0.5 * torch.Tensor([(ksz[0] + 1) % 2, (ksz[1] + 1) % 2])
            for i, T in enumerate(self.transforms[:x.shape[0]]):
                sample_center = center_pos + torch.Tensor(T.shift) / self.img_support_sz * sz
                y[i, 0, ...] = dcf.label_function_spatial(sz, sig, sample_center)
        hanning = np.array([1,0.3,0.3,1])
        self.wgt[0][:,0,...] = torch.tensor(np.outer(hanning, hanning)) # penalty coefficient of correlation filter kernel
        # Return only the ones to use for initial training
        return TensorList([y[:x.shape[0], ...] for y, x in zip(self.y, train_x)])

    def init_memory(self, train_x):
        # Initialize first-frame training samples
        self.num_init_samples = train_x.size(0)
        self.init_sample_weights = TensorList([x.new_ones(1) / x.shape[0] for x in train_x])
        self.init_training_samples = train_x

        # Sample counters and weights
        self.num_stored_samples = self.num_init_samples.copy()
        self.previous_replace_ind = [None] * len(self.num_stored_samples)
        self.sample_weights = TensorList([x.new_zeros(self.params.sample_memory_size) for x in train_x])
        for sw, init_sw, num in zip(self.sample_weights, self.init_sample_weights, self.num_init_samples):
            sw[:num] = init_sw

        # Initialize memory
        self.training_samples = TensorList(
            [x.new_zeros(self.params.sample_memory_size, cdim, x.shape[2], x.shape[3]) for x, cdim in
             zip(train_x, self.compressed_dim)])
        self.wgt = TensorList([train_x[0].new_zeros(1, 1, 4, 4)])

    def update_memory(self, sample_x: TensorList, sample_y: TensorList, learning_rate=None):
        replace_ind = self.update_sample_weights(self.sample_weights, self.previous_replace_ind,
                                                 self.num_stored_samples, self.num_init_samples, self.fparams,
                                                 learning_rate)
        self.previous_replace_ind = replace_ind
        for train_samp, x, ind in zip(self.training_samples, sample_x, replace_ind):
            train_samp[ind:ind + 1, ...] = x
        for y_memory, y, ind in zip(self.y, sample_y, replace_ind):
            y_memory[ind:ind + 1, ...] = y
        if self.hinge_mask is not None:
            for m, y, ind in zip(self.hinge_mask, sample_y, replace_ind):
                m[ind:ind + 1, ...] = (y >= self.params.hinge_threshold).float()
        self.num_stored_samples += 1

    def update_sample_weights(self, sample_weights, previous_replace_ind, num_stored_samples, num_init_samples, fparams,
                              learning_rate=None):
        # Update weights and get index to replace in memory
        replace_ind = []
        for sw, prev_ind, num_samp, num_init, fpar in zip(sample_weights, previous_replace_ind, num_stored_samples,
                                                          num_init_samples, fparams):
            lr = learning_rate
            if lr is None:
                lr = fpar.learning_rate

            init_samp_weight = getattr(fpar, 'init_samples_minimum_weight', None)
            if init_samp_weight == 0:
                init_samp_weight = None
            s_ind = 0 if init_samp_weight is None else num_init

            if num_samp == 0 or lr == 1:
                sw[:] = 0
                sw[0] = 1
                r_ind = 0
            else:
                # Get index to replace
                _, r_ind = torch.min(sw[s_ind:], 0)
                r_ind = r_ind.item() + s_ind

                # Update weights
                if prev_ind is None:
                    sw /= 1 - lr
                    sw[r_ind] = lr
                else:
                    sw[r_ind] = sw[prev_ind] / (1 - lr)

            sw /= sw.sum()
            if init_samp_weight is not None and sw[:num_init].sum() < init_samp_weight:
                sw /= init_samp_weight + sw[num_init:].sum()
                sw[:num_init] = init_samp_weight / num_init

            replace_ind.append(r_ind)

        return replace_ind

    def get_label_function(self, sample_pos, sample_scale):
        # Generate label function
        train_y = TensorList()
        target_center_norm = (self.pos - sample_pos) / (sample_scale * self.img_support_sz)
        for sig, sz, ksz in zip(self.sigma, self.feature_sz, self.kernel_size):
            center = sz * target_center_norm + 0.5 * torch.Tensor([(ksz[0] + 1) % 2, (ksz[1] + 1) % 2])
            train_y.append(dcf.label_function_spatial(sz, sig, center))
        return train_y

    def create_dist(self, width, height, cx=None, cy=None):

        if cx is None:
            cx = width / 2
        if cy is None:
            cy = height / 2

        x_ = np.linspace(1, width, width) - cx
        y_ = np.linspace(1, width, width) - cy
        X, Y = np.meshgrid(x_, y_)

        return np.sqrt(np.square(X) + np.square(Y)).astype(np.float32)

    def create_dist_gauss(self, map_sz, w, h, cx=None, cy=None, p=4, sz_weight=0.7):
        # create a square-shaped distance map with a Gaussian function which can be interpreted as a distance
        # to the given bounding box (center [cx, cy], width w, height h)
        # p is power of a Gaussian function
        # sz_weight is a weight of a bounding box size in Gaussian denominator
        if cx is None:
            cx = map_sz / 2
        if cy is None:
            cy = map_sz / 2

        x_ = np.linspace(1, map_sz, map_sz) - 1 - cx
        y_ = np.linspace(1, map_sz, map_sz) - 1 - cy
        X, Y = np.meshgrid(x_, y_)
        # 1 - is needed since we need distance-like map (not Gaussian function)
        return 1 - np.exp(-((np.power(X, p) / (sz_weight * w ** p)) + (np.power(Y, p) / (sz_weight * h ** p))))

    def init_segmentation(self, image, bb, init_mask=None):

        init_patch_crop, f_ = prutils.sample_target(image, np.array(bb), self.params.segm_search_area_factor,
                                                    output_sz=self.params.segm_output_sz)

        self.segmentation_task = False
        if init_mask is not None:
            mask = copy.deepcopy(init_mask).astype(np.float32)
            self.segmentation_task = True
            # self.params.segm_optimize_polygon = True
            # segmentation videos are shorter - therefore larger scale change factor can be used
            self.params.min_scale_change_factor = 0.9
            self.params.max_scale_change_factor = 1.1
            self.params.segm_mask_thr = 0.2
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.int32)
            if hasattr(self, 'gt_poly'):
                p1 = self.gt_poly[:2]
                p2 = self.gt_poly[2:4]
                p3 = self.gt_poly[4:6]
                p4 = self.gt_poly[6:]
                cv2.fillConvexPoly(mask, np.array([p1, p2, p3, p4], dtype=np.int32), 1)
                mask = mask.astype(np.float32)
                ###################### for visualization #########################
                '''
                if p1[0] < 0:
                   p1[0] = 0
                if p1[1] < 0:
                    p1[1] = 0
                plt.figure(figsize=(10,10))
                init_map = image[int(p1[1]):int(p3[1]+1),int(p1[0]):int(p3[0]+1),:]
                plt.xticks([])
                plt.yticks([])
                plt.imshow(init_patch_crop)
                #plt.show()
                #plt.title('init patch')
                '''
                ##################################################################
            else:
                p1 = bb[:2]
                p2 = [bb[0] + bb[2], bb[1]]
                p3 = [bb[0] + bb[2], bb[1] + bb[3]]
                p4 = [bb[0], bb[1] + bb[3]]
                cv2.fillConvexPoly(mask, np.array([p1, p2, p3, p4], dtype=np.int32), 1)
                mask = mask.astype(np.float32)

        init_mask_patch_np, patch_factor_init = prutils.sample_target(mask, np.array(bb),
                                                                      self.params.segm_search_area_factor,
                                                                      output_sz=self.params.segm_output_sz, pad_val=0)

        # network was renamed therefore we need to specify constructor_module and constructor_fun_name  self.params.segm_net_path
        segm_net, _ = load_network(self.params.segm_net_path, backbone_pretrained=False,
                                   constructor_module='ltr.models.segm.segm',
                                   constructor_fun_name='segm_resnet50')

        if self.params.use_gpu:
            segm_net.cuda()
        segm_net.eval()

        for p in segm_net.segm_predictor.parameters():
            p.requires_grad = False

        self.params.segm_normalize_mean = np.array(self.params.segm_normalize_mean).reshape((1, 1, 3))
        self.params.segm_normalize_std = np.array(self.params.segm_normalize_std).reshape((1, 1, 3))

        # normalize input image
        init_patch_norm_ = init_patch_crop.astype(np.float32) / float(255)
        init_patch_norm_ -= self.params.segm_normalize_mean
        init_patch_norm_ /= self.params.segm_normalize_std

        # create distance map for discriminative segmentation
        if self.params.segm_use_dist:
            if self.params.segm_dist_map_type == 'center':
                # center-based dist map
                dist_map = self.create_dist(init_patch_crop.shape[0], init_patch_crop.shape[1])
            elif self.params.segm_dist_map_type == 'bbox':
                # bbox-based dist map
                dist_map = self.create_dist_gauss(self.params.segm_output_sz, bb[2] * patch_factor_init,
                                                  bb[3] * patch_factor_init)
            else:
                print('Error: Unknown distance map type.')
                exit(-1)

            dist_map = torch.Tensor(dist_map)

        # put image patch and mask to GPU
        init_patch = torch.Tensor(init_patch_norm_)
        init_mask_patch = torch.Tensor(init_mask_patch_np)
        if self.params.use_gpu:
            init_patch = init_patch.to(self.params.device)
            init_mask_patch = init_mask_patch.to(self.params.device)
            if self.params.segm_use_dist:
                dist_map = dist_map.to(self.params.device)
                dist_map = torch.unsqueeze(torch.unsqueeze(dist_map, dim=0), dim=0)
                test_dist_map = [dist_map]
            else:
                test_dist_map = None

        # reshape image for the feature extractor
        init_patch = torch.unsqueeze(init_patch, dim=0).permute(0, 3, 1, 2)
        init_mask_patch = torch.unsqueeze(torch.unsqueeze(init_mask_patch, dim=0), dim=0)

        # extract features (extracting twice on the same patch - not necessary)
        train_feat = segm_net.extract_backbone_features(init_patch)

        # prepare features in the list (format for the network)
        train_feat_segm = [feat for feat in train_feat.values()]
        test_feat_segm = [feat for feat in train_feat.values()]


        train_masks = [init_mask_patch]

        if init_mask is None:
            iters = 0
            while iters < 1:
                # Obtain segmentation prediction
                segm_pred = segm_net.segm_predictor(test_feat_segm, train_feat_segm, train_masks, test_dist_map)
                 
                # softmax on the prediction (during training this is done internaly when calculating loss)
                # take only the positive channel as predicted segmentation mask
                mask = F.softmax(segm_pred, dim=1)[0, 0, :, :].cpu().numpy()
                mask_real = mask
                mask = (mask > self.params.init_segm_mask_thr).astype(np.float32)

                if hasattr(self, 'gt_poly'):
                    # dilate polygon-based mask
                    # dilate only if given mask is made from polygon, not from axis-aligned bb (since rotated bb is much tighter)
                    dil_kernel_sz = max(5, int(round(0.05 * min(self.target_sz).item() * f_)))
                    kernel = np.ones((dil_kernel_sz, dil_kernel_sz), np.uint8)
                    mask_dil = cv2.dilate(init_mask_patch_np, kernel, iterations=1)
                    mask = mask * mask_dil
                else:
                    mask = mask * init_mask_patch_np

                target_pixels = np.sum((mask > 0.5).astype(np.float32))
                self.segm_init_target_pixels = target_pixels
                ###################### for visualization #########################
                '''
                plt.figure(figsize=(10,10))
                plt.imshow(mask)
                plt.xticks([])
                plt.yticks([])
                #plt.title('maskInit')
                plt.show()
                '''
                ##################################################################
                mask_gpu = torch.unsqueeze(torch.unsqueeze(torch.tensor(mask), dim=0), dim=0).to(self.params.device)
                train_masks = [mask_gpu]
            
                iters += 1
        else:
            init_mask_patch_np = (init_mask_patch_np > 0.1).astype(np.float32)
            target_pixels = np.sum((init_mask_patch_np).astype(np.float32))
            self.segm_init_target_pixels = target_pixels

            mask_gpu = torch.unsqueeze(torch.unsqueeze(torch.tensor(init_mask_patch_np), dim=0), dim=0).to(
                self.params.device)

        # store everything that is needed for later
        self.segm_net = segm_net
        self.train_feat_segm = train_feat_segm
        self.init_mask_patch = mask_gpu
        self.segm_feat_ups = [] ## updated features of segmentation samples
        self.up_masks = []      ## updated masks of segmentation samples
        self.segm_hard_neg = False ## flag of segmentation hard-negative

        if self.params.segm_use_dist:
            self.dist_map = dist_map

        self.mask_pixels = np.array([np.sum(mask)])

    def segment_target(self, image, pos, sz):
        # pos and sz are in the image coordinates
        # construct new bounding box first
        tlx_ = pos[1] - sz[1] / 2
        tly_ = pos[0] - sz[0] / 2
        w_ = sz[1]
        h_ = sz[0]
        bb = [tlx_.item(), tly_.item(), w_.item(), h_.item()]

        # extract patch
        patch, f_ = prutils.sample_target(image, np.array(bb), self.params.segm_search_area_factor,
                                          output_sz=self.params.segm_output_sz)

        segm_crop_sz = math.ceil(math.sqrt(bb[2] * bb[3]) * self.params.segm_search_area_factor)

        # normalize input image
        init_patch_norm_ = patch.astype(np.float32) / float(255)
        init_patch_norm_ -= self.params.segm_normalize_mean
        init_patch_norm_ /= self.params.segm_normalize_std

        # put image patch and mask to GPU
        patch_gpu = torch.Tensor(init_patch_norm_)
        if self.params.use_gpu:
            patch_gpu = patch_gpu.to(self.params.device)

            # reshape image for the feature extractor
            patch_gpu = torch.unsqueeze(patch_gpu, dim=0).permute(0, 3, 1, 2)

        # extract features (extracting twice on the same patch - not necessary)
        test_feat = self.segm_net.extract_backbone_features(patch_gpu)

        # prepare features in the list (format for the network)
        test_feat_segm = [feat for feat in test_feat.values()]
        train_masks = [self.init_mask_patch]
        if self.params.segm_use_dist:
            if self.params.segm_dist_map_type == 'center':
                # center-based distance map
                test_dist_map = [self.dist_map]
            elif self.params.segm_dist_map_type == 'bbox':
                # bbox-based distance map
                D = self.create_dist_gauss(self.params.segm_output_sz, w_.item() * f_, h_.item() * f_)
                test_dist_map = [torch.unsqueeze(torch.unsqueeze(torch.Tensor(D).to(self.params.device), dim=0), dim=0)]
        else:
            test_dist_map = None

        segm_pred = self.segm_net.segm_predictor(test_feat_segm, self.train_feat_segm, train_masks, test_dist_map, self.segm_feat_ups, self.up_masks, segm_update_flag=self.segm_update_flag)

        # softmax on the prediction (during training this is done internaly when calculating loss)
        # take only the positive channel as predicted segmentation mask
        mask = F.softmax(segm_pred, dim=1)[0, 0, :, :].cpu().numpy()   
        mask = (mask > self.params.segm_mask_thr).astype(np.uint8)

        if cv2.__version__[-5] == '4':
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        else:
            _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt_area = [cv2.contourArea(cnt) for cnt in contours]

        if self.segmentation_task:
            mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(mask, contours, -1, 1, thickness=-1)

        ## found segmentation hard-negative
        self.segm_hard_neg = False
        if len(cnt_area) > 1 and len(contours) != 0 and np.max(cnt_area) > 50: 
            index = np.argmax(cnt_area)
            cnt_area_tmp = [i for i in cnt_area]
            cnt_area_tmp[index] = 0
            neg_th = cnt_area_tmp[np.argmax(cnt_area_tmp)]/cnt_area[index]
            if neg_th > 0.4:  
                self.segm_hard_neg = True         
               
        if len(cnt_area) > 0 and len(contours) != 0 and np.max(cnt_area) > 50:  
            contour = contours[np.argmax(cnt_area)]  # use max area polygon
            polygon = contour.reshape(-1, 2)

            prbox = np.reshape(cv2.boxPoints(cv2.minAreaRect(polygon)), (4, 2))  # Rotated Rectangle
            prbox_init = copy.deepcopy(prbox)

            prbox_opt = np.array([])
            if self.params.segm_optimize_polygon:
                if not self.segmentation_task:
                    mask = np.zeros(mask.shape, dtype=np.uint8)
                    cv2.drawContours(mask, [contour], -1, 1, thickness=-1)

                t_opt_start_ = time.time()
                ## rotated bounding box for VOT ##
                prbox_opt_ = fit_bbox_to_mask(mask.astype(np.int32), rotated=True)
                bbox_opt_time = time.time() - t_opt_start_
                if prbox_opt_ is not None:
                    A1 = np.linalg.norm(np.array([prbox[0, 0], prbox[0, 1]]) - np.array([prbox[1, 0], prbox[1, 1]])) * \
                         np.linalg.norm(np.array([prbox[1, 0], prbox[1, 1]]) - np.array([prbox[2, 0], prbox[2, 1]]))
                    A_new = np.linalg.norm(np.array([prbox_opt_[0, 0], prbox_opt_[0, 1]]) - np.array(
                        [prbox_opt_[1, 0], prbox_opt_[1, 1]])) * \
                            np.linalg.norm(np.array([prbox_opt_[1, 0], prbox_opt_[1, 1]]) - np.array(
                                [prbox_opt_[2, 0], prbox_opt_[2, 1]]))
                    area_ratio = A_new / A1

                    if area_ratio > 0.1 and area_ratio < 2.5:
                        prbox_opt = prbox_opt_

            displacement = np.mean(prbox, axis=0) - np.array([mask.shape[0] / 2, mask.shape[1] / 2])
            prbox = (prbox - np.mean(prbox, axis=0) + displacement) / f_ + np.array([pos[1].item(), pos[0].item()])

            if self.params.segm_scale_estimation:

                # use pixels_ratio to determine if new scale should be estimated or not
                mask_pixels_ = np.max(cnt_area)
                pixels_ratio = abs(np.mean(self.mask_pixels) - mask_pixels_) / np.mean(self.mask_pixels)
                if self.uncert_score < self.params.uncertainty_segm_scale_thr:
                    if pixels_ratio < self.params.segm_pixels_ratio: 
                        self.mask_pixels = np.append(self.mask_pixels, mask_pixels_)
                        if self.mask_pixels.size > self.params.mask_pixels_budget_sz:
                            self.mask_pixels = np.delete(self.mask_pixels, 0)

                        new_aabb = self.poly_to_aabbox(prbox[:, 0], prbox[:, 1])
                        new_target_scale = (math.sqrt(new_aabb[2] * new_aabb[3]) * self.params.search_area_scale) / \
                                           self.img_sample_sz[0]
                        rel_scale_ch = (abs(new_target_scale - self.target_scale) / self.target_scale).item()
                        if new_target_scale > self.params.segm_min_scale and rel_scale_ch < self.params.max_rel_scale_ch_thr:
                            self.target_scale = max(self.target_scale * self.params.min_scale_change_factor,
                                                    min(self.target_scale * self.params.max_scale_change_factor,
                                                        new_target_scale))

            if not self.params.segm_scale_estimation or pixels_ratio < self.params.consider_segm_pixels_ratio:
                self.pos[0] = np.mean(prbox[:, 1])
                self.pos[1] = np.mean(prbox[:, 0])

            if not self.params.segm_scale_estimation or pixels_ratio < self.params.segm_pixels_ratio:
                if prbox_opt.size > 0:
                    displacement_opt = np.mean(prbox_opt, axis=0) - np.array([mask.shape[0] / 2, mask.shape[1] / 2])
                    prbox = (prbox_opt - np.mean(prbox_opt, axis=0) + displacement_opt) / f_ + np.array(
                        [pos[1].item(), pos[0].item()])
                ## rotated bounding box ##
                pred_region_rotate = [prbox[0, 0], prbox[0, 1], prbox[1, 0], prbox[1, 1], prbox[2, 0], prbox[2, 1],
                                      prbox[3, 0], prbox[3, 1]]
                pred_region_xywh = [np.min(prbox[:, 0]) + 1, np.min(prbox[:, 1]) + 1,
                                    np.max(prbox[:, 0]) - np.min(prbox[:, 0]) + 1,
                                    np.max(prbox[:, 1]) - np.min(prbox[:, 1]) + 1]
                ## for visualization ##
                '''
                if True:
                    #pdb.set_trace()
                    pred_region1 = [prbox[0, 0], prbox[0, 1], prbox[1, 0], prbox[1, 1], prbox[2, 0], prbox[2, 1],prbox[3, 0], prbox[3, 1]]
                    p1=(round(pred_region1[0]),round(pred_region1[1]))
                    p2=(round(pred_region1[2]),round(pred_region1[3]))
                    p3=(round(pred_region1[4]),round(pred_region1[5]))
                    p4=(round(pred_region1[6]),round(pred_region1[7]))
                    im1 = image
                    cv2.line(im1,p1,p2,(255,0,0),2)
                    cv2.line(im1,p2,p3,(255,0,0),2)
                    cv2.line(im1,p3,p4,(255,0,0),2)
                    cv2.line(im1,p4,p1,(255,0,0),2)
                    patch1, f1_ = prutils.sample_target(im1, np.array(bb), self.params.segm_search_area_factor,output_sz=self.params.segm_output_sz)
                    plt.figure(figsize=(10,10))
                    #plt.imshow(patch1)
                    #plt.figure(figsize=(10,10))
                    #plt.imshow(mask)
                    patch1[:,:,0] = patch1[:,:,0]*(1-mask) + patch1[:,:,0]*mask*0.5
                    patch1[:,:,1] = patch1[:,:,1]*(1-mask) + patch1[:,:,1]*mask*0.5+mask*100
                    patch1[:,:,2] = patch1[:,:,2]*(1-mask) + patch1[:,:,2]*mask*0.5
                    plt.imshow(patch1)
                    plt.xticks([])
                    plt.yticks([])
                    #plt.title('tracking_results')
                    plt.show()
                '''
                #######################
                return pred_region_xywh, pred_region_rotate

        return None, None

    def poly_to_aabbox(self, x_, y_):
        # keep the center and area of the polygon
        # change aspect ratio of the original bbox
        cx = np.mean(x_)
        cy = np.mean(y_)
        x1 = np.min(x_)
        x2 = np.max(x_)
        y1 = np.min(y_)
        y2 = np.max(y_)
        A1 = np.linalg.norm(np.array([x_[0], y_[0]]) - np.array([x_[1], y_[1]])) * \
             np.linalg.norm(np.array([x_[1], y_[1]]) - np.array([x_[2], y_[2]]))
        A2 = (x2 - x1) * (y2 - y1)
        s = np.sqrt(A1 / A2)
        w = s * (x2 - x1) + 1
        h = s * (y2 - y1) + 1
        return np.array([cx - w / 2, cy - h / 2, w, h])

    ## IouNet ##
    def get_sample_location(self, sample_coord):
        """Get the location of the extracted sample."""
        sample_coord = sample_coord.float()
        sample_pos = 0.5*(sample_coord[:,:2] + sample_coord[:,2:] - 1)
        sample_scales = ((sample_coord[:,2:] - sample_coord[:,:2]) / self.img_sample_sz1).prod(dim=1).sqrt()
        return sample_pos, sample_scales

    def get_centered_sample_pos(self):
        """Get the center position for the new sample. Make sure the target is correctly centered."""
        return self.pos + ((self.feature_sz1 + self.kernel_size1) % 2) * self.target_scale1 * \
               self.img_support_sz1 / (2*self.feature_sz1)

    def extract_backbone_features(self, im: torch.Tensor, pos: torch.Tensor, scales, sz: torch.Tensor):
        im_patches, patch_coords = sample_patch_multiscale(im, pos, scales, sz,
                                                           mode=self.params.get('border_mode', 'replicate'),
                                                           max_scale_change=self.params.get('patch_max_scale_change', None))
        with torch.no_grad():
            backbone_feat = self.net.extract_backbone(im_patches)
        return backbone_feat, patch_coords, im_patches

    def get_iou_backbone_features(self, backbone_feat):
        return self.net.get_backbone_bbreg_feat(backbone_feat)

    def get_iou_features(self, backbone_feat):
        with torch.no_grad():
            return self.net.bb_regressor.get_iou_feat(self.get_iou_backbone_features(backbone_feat))

    def get_iou_modulation(self, iou_backbone_feat, target_boxes):
        with torch.no_grad():
            return self.net.bb_regressor.get_modulation(iou_backbone_feat, target_boxes)


    def generate_init_samples_IouNet(self, im: torch.Tensor) -> TensorList:
        """Perform data augmentation to generate initial training samples."""

        mode = self.params.get('border_mode', 'replicate')
        if mode == 'inside':
            # Get new sample size if forced inside the image
            im_sz = torch.Tensor([im.shape[2], im.shape[3]])
            sample_sz = self.target_scale1 * self.img_sample_sz1
            shrink_factor = (sample_sz.float() / im_sz)
            if mode == 'inside':
                shrink_factor = shrink_factor.max()
            elif mode == 'inside_major':
                shrink_factor = shrink_factor.min()
            shrink_factor.clamp_(min=1, max=self.params.get('patch_max_scale_change', None))
            sample_sz = (sample_sz.float() / shrink_factor)
            self.init_sample_scale = (sample_sz / self.img_sample_sz1).prod().sqrt()
            tl = self.pos - (sample_sz - 1) / 2
            br = self.pos + sample_sz / 2 + 1
            global_shift = - ((-tl).clamp(0) - (br - im_sz).clamp(0)) / self.init_sample_scale
        else:
            self.init_sample_scale = self.target_scale1
            global_shift = torch.zeros(2)

        self.init_sample_pos = self.pos.round()

        # Compute augmentation size
        aug_expansion_factor = self.params.get('augmentation_expansion_factor', None)
        aug_expansion_sz = self.img_sample_sz1.clone()
        aug_output_sz = None
        if aug_expansion_factor is not None and aug_expansion_factor != 1:
            aug_expansion_sz = (self.img_sample_sz1 * aug_expansion_factor).long()
            aug_expansion_sz += (aug_expansion_sz - self.img_sample_sz1.long()) % 2
            aug_expansion_sz = aug_expansion_sz.float()
            aug_output_sz = self.img_sample_sz1.long().tolist()

        # Random shift for each sample
        get_rand_shift = lambda: None
        random_shift_factor = self.params.get('random_shift_factor1', 0)
        if random_shift_factor > 0:
            get_rand_shift = lambda: ((torch.rand(2) - 0.5) * self.img_sample_sz1 * random_shift_factor + global_shift).long().tolist()

        # Always put identity transformation first, since it is the unaugmented sample that is always used
        self.transforms1 = [augmentation1.Identity(aug_output_sz, global_shift.long().tolist())]

        augs = self.params.augmentation1 if self.params.get('use_augmentation', True) else {}

        # Add all augmentations
        if 'shift' in augs:
            self.transforms1.extend([augmentation1.Translation(shift, aug_output_sz, global_shift.long().tolist()) for shift in augs['shift']])
        if 'relativeshift' in augs:
            get_absolute = lambda shift: (torch.Tensor(shift) * self.img_sample_sz1/2).long().tolist()
            self.transforms1.extend([augmentation1.Translation(get_absolute(shift), aug_output_sz, global_shift.long().tolist()) for shift in augs['relativeshift']])
        if 'fliplr' in augs and augs['fliplr']:
            self.transforms1.append(augmentation1.FlipHorizontal(aug_output_sz, get_rand_shift()))
        if 'blur' in augs:
            self.transforms1.extend([augmentation1.Blur(sigma, aug_output_sz, get_rand_shift()) for sigma in augs['blur']])
        if 'scale' in augs:
            self.transforms1.extend([augmentation1.Scale(scale_factor, aug_output_sz, get_rand_shift()) for scale_factor in augs['scale']])
        if 'rotate' in augs:
            self.transforms1.extend([augmentation1.Rotate(angle, aug_output_sz, get_rand_shift()) for angle in augs['rotate']])

        # Extract augmented image patches
        im_patches = sample_patch_transformed(im, self.init_sample_pos, self.init_sample_scale, aug_expansion_sz, self.transforms1)

        # Extract initial backbone features
        with torch.no_grad():
            init_backbone_feat = self.net.extract_backbone(im_patches)

        return init_backbone_feat

    def init_target_boxes(self):
        """Get the target bounding boxes for the initial augmented samples."""
        self.classifier_target_box = self.get_iounet_box(self.pos, self.target_sz1, self.init_sample_pos, self.init_sample_scale)
        init_target_boxes = TensorList()
        for T in self.transforms1:
            init_target_boxes.append(self.classifier_target_box + torch.Tensor([T.shift[1], T.shift[0], 0, 0]))
        init_target_boxes = torch.cat(init_target_boxes.view(1, 4), 0).to(self.params.device)
        self.target_boxes = init_target_boxes.new_zeros(self.params.sample_memory_size1, 4)
        self.target_boxes[:init_target_boxes.shape[0],:] = init_target_boxes
        return init_target_boxes

    def update_state(self, new_pos, new_scale = None):
        # Update scale
        if new_scale is not None:
            self.target_scale = new_scale.clamp(self.min_scale_factor, self.max_scale_factor)
            self.target_sz = self.base_target_sz * self.target_scale
            self.target_sz1 = self.target_sz.clone()

        # Update pos
        inside_ratio = self.params.get('target_inside_ratio', 0.2)
        inside_offset = (inside_ratio - 0.5) * self.target_sz
        self.pos = torch.max(torch.min(new_pos, self.image_sz - inside_offset), inside_offset)


    def get_iounet_box(self, pos, sz, sample_pos, sample_scale):
        """All inputs in original image coordinates.
        Generates a box in the cropped image sample reference frame, in the format used by the IoUNet."""
        box_center = (pos - sample_pos) / sample_scale + (self.img_sample_sz1 - 1) / 2
        box_sz = sz / sample_scale
        target_ul = box_center - (box_sz - 1) / 2
        return torch.cat([target_ul.flip((0,)), box_sz.flip((0,))])


    def init_iou_net(self, backbone_feat):
        # Setup IoU net and objective
        for p in self.net.bb_regressor.parameters():
            p.requires_grad = False

        # Get target boxes for the different augmentations
        self.classifier_target_box = self.get_iounet_box(self.pos, self.target_sz1, self.init_sample_pos, self.init_sample_scale)
        target_boxes = TensorList()
        if self.params.iounet_augmentation:
            for T in self.transforms1:
                if not isinstance(T, (augmentation1.Identity, augmentation1.Translation, augmentation1.FlipHorizontal, augmentation1.FlipVertical, augmentation1.Blur)):
                    break
                target_boxes.append(self.classifier_target_box + torch.Tensor([T.shift[1], T.shift[0], 0, 0]))
        else:
            target_boxes.append(self.classifier_target_box + torch.Tensor([self.transforms1[0].shift[1], self.transforms1[0].shift[0], 0, 0]))
        target_boxes = torch.cat(target_boxes.view(1,4), 0).to(self.params.device)

        # Get iou features
        iou_backbone_feat = self.get_iou_backbone_features(backbone_feat)

        # Remove other augmentations such as rotation
        iou_backbone_feat = TensorList([x[:target_boxes.shape[0],...] for x in iou_backbone_feat])

        # Get modulation vector
        self.iou_modulation = self.get_iou_modulation(iou_backbone_feat, target_boxes)
        if torch.is_tensor(self.iou_modulation[0]):
            self.iou_modulation = TensorList([x.detach().mean(0) for x in self.iou_modulation])

    def refine_target_box(self, backbone_feat, sample_pos, sample_scale, scale_ind, update_scale = True):
        """Run the ATOM IoUNet to refine the target bounding box."""

        if hasattr(self.net.bb_regressor, 'predict_bb'):
            return self.direct_box_regression(backbone_feat, sample_pos, sample_scale, scale_ind, update_scale)

        # Initial box for refinement
        init_box = self.get_iounet_box(self.pos, self.target_sz1, sample_pos, sample_scale)

        # Extract features from the relevant scale
        iou_features = self.get_iou_features(backbone_feat)
        iou_features = TensorList([x[scale_ind:scale_ind+1,...] for x in iou_features])

        # Generate random initial boxes
        init_boxes = init_box.view(1,4).clone()
        if self.params.num_init_random_boxes > 0:
            square_box_sz = init_box[2:].prod().sqrt()
            rand_factor = square_box_sz * torch.cat([self.params.box_jitter_pos * torch.ones(2), self.params.box_jitter_sz * torch.ones(2)])

            minimal_edge_size = init_box[2:].min()/3
            rand_bb = (torch.rand(self.params.num_init_random_boxes, 4) - 0.5) * rand_factor
            new_sz = (init_box[2:] + rand_bb[:,2:]).clamp(minimal_edge_size)
            new_center = (init_box[:2] + init_box[2:]/2) + rand_bb[:,:2]
            init_boxes = torch.cat([new_center - new_sz/2, new_sz], 1)
            init_boxes = torch.cat([init_box.view(1,4), init_boxes])

        # Optimize the boxes
        output_boxes, output_iou = self.optimize_boxes(iou_features, init_boxes)

        # Remove weird boxes
        output_boxes[:, 2:].clamp_(1)
        aspect_ratio = output_boxes[:,2] / output_boxes[:,3]
        keep_ind = (aspect_ratio < self.params.maximal_aspect_ratio) * (aspect_ratio > 1/self.params.maximal_aspect_ratio)
        output_boxes = output_boxes[keep_ind,:]
        output_iou = output_iou[keep_ind]

        # If no box found
        if output_boxes.shape[0] == 0:
            return

        # Predict box
        k = self.params.get('iounet_k', 5)
        topk = min(k, output_boxes.shape[0])
        _, inds = torch.topk(output_iou, topk)
        predicted_box = output_boxes[inds, :].mean(0)
        predicted_iou = output_iou.view(-1, 1)[inds, :].mean(0)

        # Get new position and size
        new_pos = predicted_box[:2] + predicted_box[2:] / 2
        new_pos = (new_pos.flip((0,)) - (self.img_sample_sz1 - 1) / 2) * sample_scale + sample_pos
        new_target_sz = predicted_box[2:].flip((0,)) * sample_scale
        new_scale = torch.sqrt(new_target_sz.prod() / self.base_target_sz1.prod())

        self.pos_iounet = new_pos.clone()
        self.target_sz_iounet = new_target_sz

        # self.visualize_iou_pred(iou_features, predicted_box)


    def optimize_boxes(self, iou_features, init_boxes):
        box_refinement_space = self.params.get('box_refinement_space', 'default')
        if box_refinement_space == 'default':
            return self.optimize_boxes_default(iou_features, init_boxes)
        if box_refinement_space == 'relative':
            return self.optimize_boxes_relative(iou_features, init_boxes)
        raise ValueError('Unknown box_refinement_space {}'.format(box_refinement_space))


    def optimize_boxes_default(self, iou_features, init_boxes):
        """Optimize iounet boxes with the default parametrization"""
        output_boxes = init_boxes.view(1, -1, 4).to(self.params.device)
        step_length = self.params.box_refinement_step_length
        if isinstance(step_length, (tuple, list)):
            step_length = torch.Tensor([step_length[0], step_length[0], step_length[1], step_length[1]], device=self.params.device).view(1,1,4)

        for i_ in range(self.params.box_refinement_iter):
            # forward pass
            bb_init = output_boxes.clone().detach()
            bb_init.requires_grad = True

            outputs = self.net.bb_regressor.predict_iou(self.iou_modulation, iou_features, bb_init)

            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            outputs.backward(gradient = torch.ones_like(outputs))

            # Update proposal
            output_boxes = bb_init + step_length * bb_init.grad * bb_init[:, :, 2:].repeat(1, 1, 2)
            output_boxes.detach_()

            step_length *= self.params.box_refinement_step_decay

        return output_boxes.view(-1,4).cpu(), outputs.detach().view(-1).cpu()


    def optimize_boxes_relative(self, iou_features, init_boxes):
        output_boxes = init_boxes.view(1, -1, 4).to(self.params.device)
        step_length = self.params.box_refinement_step_length
        if isinstance(step_length, (tuple, list)):
            step_length = torch.Tensor([step_length[0], step_length[0], step_length[1], step_length[1]]).to(self.params.device).view(1,1,4)

        sz_norm = output_boxes[:,:1,2:].clone()
        output_boxes_rel = bbutils.rect_to_rel(output_boxes, sz_norm)
        for i_ in range(self.params.box_refinement_iter):
            # forward pass
            bb_init_rel = output_boxes_rel.clone().detach()
            bb_init_rel.requires_grad = True

            bb_init = bbutils.rel_to_rect(bb_init_rel, sz_norm)
            outputs = self.net.bb_regressor.predict_iou(self.iou_modulation, iou_features, bb_init)

            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            outputs.backward(gradient = torch.ones_like(outputs))

            # Update proposal
            output_boxes_rel = bb_init_rel + step_length * bb_init_rel.grad
            output_boxes_rel.detach_()

            step_length *= self.params.box_refinement_step_decay

        output_boxes = bbutils.rel_to_rect(output_boxes_rel, sz_norm)

        return output_boxes.view(-1,4).cpu(), outputs.detach().view(-1).cpu()

    def direct_box_regression(self, backbone_feat, sample_pos, sample_scale, scale_ind, update_scale = True):
        """Implementation of direct bounding box regression."""

        # Initial box for refinement
        init_box = self.get_iounet_box(self.pos, self.target_sz1, sample_pos, sample_scale)

        # Extract features from the relevant scale
        iou_features = self.get_iou_features(backbone_feat)
        iou_features = TensorList([x[scale_ind:scale_ind+1,...] for x in iou_features])

        # Generate random initial boxes
        init_boxes = init_box.view(1, 1, 4).clone().to(self.params.device)

        # Optimize the boxes
        output_boxes = self.net.bb_regressor.predict_bb(self.iou_modulation, iou_features, init_boxes).view(-1,4).cpu()

        # Remove weird boxes
        output_boxes[:, 2:].clamp_(1)

        predicted_box = output_boxes[0, :]

        # Get new position and size
        new_pos = predicted_box[:2] + predicted_box[2:] / 2
        new_pos = (new_pos.flip((0,)) - (self.img_sample_sz1 - 1) / 2) * sample_scale + sample_pos
        new_target_sz = predicted_box[2:].flip((0,)) * sample_scale
        new_scale_bbr = torch.sqrt(new_target_sz.prod() / self.base_target_sz1.prod())
        new_scale = new_scale_bbr

        self.pos_iounet = new_pos.clone()
        self.target_sz_iounet = new_target_sz


    def visualize_iou_pred(self, iou_features, center_box):
        center_box = center_box.view(1,1,4)
        sz_norm = center_box[...,2:].clone()
        center_box_rel = bbutils.rect_to_rel(center_box, sz_norm)

        pos_dist = 1.0
        sz_dist = math.log(3.0)
        pos_step = 0.01
        sz_step = 0.01

        pos_scale = torch.arange(-pos_dist, pos_dist+pos_step, step=pos_step)
        sz_scale = torch.arange(-sz_dist, sz_dist+sz_step, step=sz_step)

        bbx = torch.zeros(1, pos_scale.numel(), 4)
        bbx[0,:,0] = pos_scale.clone()
        bby = torch.zeros(pos_scale.numel(), 1, 4)
        bby[:,0,1] = pos_scale.clone()
        bbw = torch.zeros(1, sz_scale.numel(), 4)
        bbw[0,:,2] = sz_scale.clone()
        bbh = torch.zeros(sz_scale.numel(), 1, 4)
        bbh[:,0,3] = sz_scale.clone()

        pos_boxes = bbutils.rel_to_rect((center_box_rel + bbx) + bby, sz_norm).view(1,-1,4).to(self.params.device)
        sz_boxes = bbutils.rel_to_rect((center_box_rel + bbw) + bbh, sz_norm).view(1,-1,4).to(self.params.device)

        pos_scores = self.net.bb_regressor.predict_iou(self.iou_modulation, iou_features, pos_boxes).exp()
        sz_scores = self.net.bb_regressor.predict_iou(self.iou_modulation, iou_features, sz_boxes).exp()

        show_tensor(pos_scores.view(pos_scale.numel(),-1), title='Position scores', fig_num=21)
        show_tensor(sz_scores.view(sz_scale.numel(),-1), title='Size scores', fig_num=22)


    def visdom_draw_tracking(self, image, box, segmentation=None):
        if hasattr(self, 'search_area_box'):
            self.visdom.register((image, box, self.search_area_box), 'Tracking', 1, 'Tracking')
        else:
            self.visdom.register((image, box), 'Tracking', 1, 'Tracking')
