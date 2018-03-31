In this post we will understand and implement the components of the modern object detection model Faster-RCNN. Object detectors involve many parts and
it can be difficult to follow the code in the open implementations available. Here we will give a clear layout of of such a model. This post is divided into two parts. In this first one, we will construct the Faster-RCNN network and several of it's components. This will allow us to perform inference using the pretrained weights available from <a href="https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md">Facebook's Detectron</a>, which will help us understand how that object-detection framework works. In the  second part of the post, we will actually train the network ourselves and understand the steps involved in that process. 

Here we will take certain components for granted - NMS, bounding box transformations, ROI-Align, and the proposal layer. For an explanation of how these work please consult the papers and the <a href='https://github.com/youssefzaky/mask_rcnn'>full code repository</a>. The focus here is more on how the pieces fit together.

<h3> The Two Stages </h3>

The Faster-RCNN family of detectors works in two stages. The first stage, the Region Proposal Network (RPN), outputs box regions and their associated 'objectness' score (i.e., object vs no object). These proposals are filtered, then used to crop features from the top-level of the backbone feature extractor (e.g., Resnet-50). This process of feature cropping was done by ROI-pooling in the original Faster-RCNN, or more recently, using ROI-Align in Mask-RCNN. The second stage, the Faster-RCNN network, takes these cropped features and refines the initial proposals of RPN, along with predicting the probability of the object class. The figure below from the Faster-RCNN paper illustrates this two-stage process:

![FRCNN](/images/frcnn/frcnn.png){: .center-image }

In the Faster-RCNN version with the Resnet feature extractor (or backbone), RPN operates on the final convolutional layer of the C4 block [3]. ROI-align is performed on the C4 features and those pooled features are first fed through the C5 block and the average pool of Resnet. These last two operations replace the fully-connected layers of VGG in the original application of Resnet to object-detection [3]. After that, the Faster-RCNN network predicts bounding-boxes and classes.

The code below show the structure of this computation and the backbone Resnet50 network (adapted from torchvision):


```python
class FasterRCNN(nn.Module):
    def __init__(self, rpn, backbone, roi_align, n_classes):
        super(FasterRCNN, self).__init__()
        self.backbone = backbone  # eg. Resnet-50
        self.rpn = rpn  # Region Proposal Network
        self.roi_align = roi_align  # ROI-Align layer
        self.bbox_pred = nn.Linear(2048, 4 * n_classes)  # bounding-box head
        self.cls_score = nn.Linear(2048, n_classes)  # class logits head

    def forward(self, images, h, w, im_scale):
	# compute block C4 features for image
        features = self.backbone(images)  
	# apply RPN to get proposals
        proposals, scores = self.rpn(features, h, w, im_scale)  
	# apply ROI align on the C4 features using the proposals
        pooled_feat = self.roi_align(features, proposals) 
	# apply the C5 block and Average pool of Resnet	  
        pooled_feat = self.backbone.top(pooled_feat).squeeze()
	# apply bounding-box head
        bbox_pred = self.bbox_pred(pooled_feat)  
	# apply class-score head
        cls_score = self.cls_score(pooled_feat)  
	# softmax to get object-class probabilities
        cls_prob = F.softmax(cls_score, dim=1)  

        return {'bbox_pred': bbox_pred, 'cls_prob': cls_prob, 'rois': proposals}
    

class BackBone(nn.Module):
    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3]):
        self.inplanes = 64
        super(BackBone, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # differences between pytorch and caffe2 resnet
        self.layer2[0].conv1.stride = (2, 2)
        self.layer2[0].conv2.stride = (1, 1)
        self.layer3[0].conv1.stride = (2, 2)
        self.layer3[0].conv2.stride = (1, 1)
        self.layer4[0].conv1.stride = (2, 2)
        self.layer4[0].conv2.stride = (1, 1)

        self.avgpool = nn.AvgPool2d(7)
        self.top = nn.Sequential(self.layer4, self.avgpool)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # forward only to layer 3
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return self.layer3(x)
```

Note that the C5 block and the Average Pool of Resnet produce a 2048-dimensional vector per ROI. The bounding-box heads and class-score heads are fully-connected layers that operate on those ROI vectors. The class-score head output dimension is the number of object classes. The bounding-box head output dimension is 4 times the number of classes. This is because a bounding box is predicted separately for each object class, later we take the box corresponding to the class with the highest score.

Now that we see the overal structure of the networks, let's see how each part works in more detail.

<h3> The Region Proposal Network </h3>

RPN works on the top convolutional layer of the C4 block. First it applies a convolution with a 3x3 kernel to get another feature map with 1024 output channels. Then two convolutional layers with 1x1 kernels are applied to get proposals and scores <b>per cell</b> in the feature map, as shown in the figure below from the paper [1]:

![RPN](/images/frcnn/rpn.png){: .center-image }

The output channels for the scores convolutional head give the 'objectness' score for each anchor at that cell. The output channels for the proposal convolutional head give the bounding boxes for each anchor at that cell.


```python
class RPN(nn.Module):
    def __init__(self, in_channels, out_channels, n_anchors, proposal_layer):
        super(RPN, self).__init__()

        self.n_anchors = n_anchors

        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bbox_pred = nn.Conv2d(out_channels, self.n_anchors * 4, 1)
        self.cls_score = nn.Conv2d(out_channels, self.n_anchors, 1)
        self.proposal_layer = proposal_layer

    def forward(self, feature_map, h, w, im_scale):
        x = self.conv(feature_map)
        x = F.relu(x, inplace=True)
        cls_score = self.cls_score(x)
        cls_prob = F.sigmoid(cls_score)
        bbox_pred = self.bbox_pred(x)
        proposals, scores = self.proposal_layer(cls_prob, bbox_pred,
                                                h, w, im_scale)
        return proposals, scores
```

Note at the end of the forward method we make a call to a proposal layer. This layer performs additional computations and filtering on the given proposals before they are passed further on. This because the outputs of the RPN need to be applied to the anchors at every cell to get the actual proposal, and the proposals need to be filtered to make sense and reduce their huge number. The steps it does are:

<ol> <li> for each location on the feature map grid grid: 
    <ul> <li> generate the anchor boxes centered on cell i </li>
        <li> apply predicted bbox deltas to each of the A anchors at cell i </li>
    </ul> </li>
    <li> clip predicted boxes to image </li>
    <li> remove predicted boxes that are smaller than a threshold </li>
    <li> sort all proposals by score from highest to lowest </li> 
    <li> take the top proposals before NMS </li>
    <li> apply NMS with a loose threshold (0.7) to the remaining proposals </li>
    <li> take top proposals after NMS </li>
    <li> return the top proposals </li>
</ol>

Below we show the forward method of the proposal layer, where these computations take place:


```python
def forward(self, rpn_cls_probs, rpn_bbox_pred, im_height, im_width, scaling_factor):
    # 1. get anchors at all features positions
    all_anchors_np = self.get_all_anchors(num_images=rpn_cls_probs.shape[0],
                                          feature_height=rpn_cls_probs.shape[2],
                                          feature_width=rpn_cls_probs.shape[3])

    all_anchors = Variable(torch.FloatTensor(all_anchors_np))
    if rpn_cls_probs.is_cuda:
        all_anchors = all_anchors.cuda()

    # Transpose and reshape predicted bbox transformations to get them
    # into the same order as the anchors:
    #   - bbox deltas will be (4 * A, H, W) format from conv output
    #   - transpose to (H, W, 4 * A)
    #   - reshape to (H * W * A, 4) where rows are ordered by (H, W, A)
    #     in slowest to fastest order to match the enumerated anchors
    bbox_deltas = rpn_bbox_pred.squeeze(0).permute(1, 2, 0).contiguous().view(-1, 4)

    # Same story for the scores:
    #   - scores are (A, H, W) format from conv output
    #   - transpose to (H, W, A)
    #   - reshape to (H * W * A, 1) where rows are ordered by (H, W, A)
    #     to match the order of anchors and bbox_deltas
    scores = rpn_cls_probs.squeeze(0).permute(1, 2, 0).contiguous().view(-1, 1)
    scores_np = scores.cpu().data.numpy()

    # 4. sort all (proposal, score) pairs by score from highest to lowest
    # 5. take top pre_nms_topN (e.g. 6000)
    # Avoid sorting possibly large arrays; First partition to get top K
    # unsorted and then sort just those (~20x faster for 200k scores)
    inds = np.argpartition(-scores_np.squeeze(), self.rpn_pre_nms_top_n)[:self.rpn_pre_nms_top_n]
    order = np.argsort(-scores_np[inds].squeeze())
    order = inds[order]

    bbox_deltas = bbox_deltas[order, :]
    scores = scores[order, :]
    scores_np = scores_np[order, :]
    all_anchors = all_anchors[order, :]

    # Transform anchors into proposals via bbox transformations
    proposals = self.bbox_transform(all_anchors, bbox_deltas, (1.0, 1.0, 1.0, 1.0))

    # 2. clip proposals to image (may result in proposals with zero area
    # that will be removed in the next step)
    proposals = self.clip_tiled_boxes(proposals, im_height, im_width)
    proposals_np = proposals.cpu().data.numpy()

    # 3. remove predicted boxes with either height or width < min_size
    keep = self.filter_boxes(proposals_np, self.rpn_min_size, scaling_factor, im_height, im_width)

    proposals = proposals[keep, :]
    proposals_np = proposals_np[keep, :]
    scores = scores[keep, :]
    scores_np = scores_np[keep]

    # 6. apply loose nms (e.g. threshold = 0.7)
    # 7. take after_nms_topN (e.g. 300)
    # 8. return the top proposals (-> RoIs top)
    keep = box_utils.nms(np.hstack((proposals_np, scores_np)), self.rpn_nms_thresh)
    keep = keep[:self.rpn_post_nms_top_n]

    proposals = proposals[keep, :]
    scores = scores[keep, :]

    return proposals, scores
```

<h3> A Complete Example </h3>

To put things together, we will do a complete example to illustrate the steps in a Faster-RCNN detection.

First we define some configuration parameters:


```python
# Number of top scoring boxes to keep before apply NMS to RPN proposals
RPN_PRE_NMS_TOP_N = 6000
# Number of top scoring boxes to keep after applying NMS to RPN proposals
RPN_POST_NMS_TOP_N = 1000
# NMS threshold used on RPN proposals
RPN_NMS_THRESH = 0.7
# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
RPN_MIN_SIZE = 0
# Size of the pooled region after RoI pooling
POOLING_SIZE = 14

TEST_NMS = 0.5
TEST_MAX_SIZE = 1333
PIXEL_MEANS = np.array([122.7717, 115.9465, 102.9801])

anchor_sizes, anchor_ratios = [32, 64, 128, 256, 512], [0.5, 1, 2]
feat_stride = 16
```

Then prepare an image for the feedforward pass (i.e, subtracting the pixel means and making sure it has a minimum dimension):


```python
x = cv2.imread('samples/15673749081_767a7fa63a_k.jpg')[:, :, ::-1]

blobs, im_scales = prep_im_for_blob(x, PIXEL_MEANS, target_sizes=(800,), max_size=TEST_MAX_SIZE)
blobs = im_list_to_blob(blobs)
img = Variable(torch.from_numpy(blobs))

im_info = torch.from_numpy(np.array([[blobs.shape[2], blobs.shape[3], im_scales[0]]], dtype=np.float32))
im_size, im_scale = [blobs.shape[2], blobs.shape[3]], im_scales[0]
```

Make the modules we need and load pretrained weights from detectron:


```python
backbone = BackBone()
proposal_layer = ProposalLayer(feat_stride, anchor_sizes, anchor_ratios,
                               RPN_PRE_NMS_TOP_N, RPN_POST_NMS_TOP_N, RPN_NMS_THRESH, RPN_MIN_SIZE)
roi_align = RoIAlign(POOLING_SIZE, POOLING_SIZE, spatial_scale=1./16.)

rpn = RPN(1024, 1024, 15, proposal_layer)
frcnn = FasterRCNN(rpn, backbone, roi_align, 81)
frcnn.load_pretrained_weights('model_final.pkl', 'resnet50_mapping.npy')

frcnn = frcnn.cuda()
frcnn.eval()
img = img.cuda()
```

Finally we do a feedforward pass, post-process the output and visualize:


```python
output = frcnn(img, im_size[0], im_size[1], im_scale)
class_scores, bbox_deltas, rois = output['cls_prob'], output['bbox_pred'], output['rois']

scores_final, boxes_final, boxes_per_class = postprocess_output(rois, im_scale, im_size, class_scores, bbox_deltas,
                                                                bbox_reg_weights=(10.0, 10.0, 5.0, 5.0))


vis.vis_one_image(
    x, 
    'output',
    'samples/',
    boxes_per_class,
    dataset=None,
    box_alpha=0.3,
    show_class=True,
    thresh=0.7,
    ext='jpg'
)
```

![Results](/images/frcnn/output.png){: .center-image }


<h3> Conclusions </h3>

Hopefully this post has shown the important pieces that go into an object detector. We did not explain some components in detail like NMS or ROI-Align, but these are important so one should understand them from the papers and the code. 

<h3> References </h3>

<ol> 
     <li> Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster r-cnn: Towards real-time object detection with region proposal networks. In Advances in neural information processing systems (pp. 91-99). </li>
     <li>He, K., Gkioxari, G., Dollár, P., & Girshick, R. (2017, October). Mask r-cnn. In Computer Vision (ICCV), 2017 IEEE International Conference on (pp. 2980-2988). IEEE.</li>
         <li>  He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778). </li>
</ol>
