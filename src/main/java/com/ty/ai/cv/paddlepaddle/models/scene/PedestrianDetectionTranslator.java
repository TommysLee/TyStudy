package com.ty.ai.cv.paddlepaddle.models.scene;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.TranslatorContext;
import com.google.common.collect.Lists;

import java.util.List;

/**
 * Pedestrian Detection Translator
 *
 * @Author Tommy
 * @Date 2024/5/5
 */
public class PedestrianDetectionTranslator implements NoBatchifyTranslator<Image, DetectedObjects> {

    /**
     * 数据前置处理
     *
     * @param ctx the toolkit for creating the input NDArray
     * @param input the input object
     * @return NDList
     * @throws Exception
     */
    @Override
    public NDList processInput(TranslatorContext ctx, Image input) throws Exception {
        NDArray array = input.toNDArray(ctx.getNDManager(), Image.Flag.COLOR);
        array = NDImageUtils.resize(array, 608, 608);
        if (!array.getDataType().equals(DataType.FLOAT32)) {
            array = array.toType(DataType.FLOAT32, false);
        }
        array = array.div(255f);
        NDArray mean = ctx.getNDManager().create(new float[] {0.485f, 0.456f, 0.406f}, new Shape(1, 1, 3));
        NDArray std = ctx.getNDManager().create(new float[] {0.229f, 0.224f, 0.225f}, new Shape(1, 1, 3));
        array = array.sub(mean);
        array = array.div(std);

        array = array.transpose(2, 0, 1); // HWC -> CHW RGB
        array = array.expandDims(0);

        int width = input.getWidth();
        int height = input.getHeight();
        NDArray imageSize = ctx.getNDManager().create(new int[] {height, width});
        imageSize = imageSize.toType(DataType.INT32, false);
        imageSize = imageSize.expandDims(0);

        return new NDList(array, imageSize);
    }

    /**
     * 数据后置处理
     *
     * @param ctx the toolkit used for post-processing
     * @param list the output NDList after inference, usually immutable in engines like
     *     PyTorch. @see <a href="https://github.com/deepjavalibrary/djl/issues/1774">Issue 1774</a>
     * @return DetectedObjects
     * @throws Exception
     */
    @Override
    public DetectedObjects processOutput(TranslatorContext ctx, NDList list) throws Exception {
        NDArray result = list.singletonOrThrow();
        float[] probabilities = result.get(":,1").toFloatArray();

        List<String> names = Lists.newArrayListWithCapacity(probabilities.length);
        List<Double> prob = Lists.newArrayListWithCapacity(probabilities.length);
        List<BoundingBox> boxes = Lists.newArrayListWithCapacity(probabilities.length);
        for (int i = 0; i < probabilities.length; i++) {
            float[] array = result.get(i).toFloatArray();
            //        [  0.          0.9627503 172.78745    22.62915   420.2703    919.949    ]
            //        [  0.          0.8364255 497.77234   161.08307   594.4088    480.63745  ]
            //        [  0.          0.7247823  94.354065  177.53668   169.24417   429.2456   ]
            //        [  0.          0.5549363  18.81821   209.29712   116.40645   471.8595   ]
            // names.add(Float.valueOf(array[0]).toString());
            names.add("行人");
            // array[0] category_id
            // array[1] confidence
            // bbox
            // array[2]
            // array[3]
            // array[4]
            // array[5]
            prob.add((double) probabilities[i]);
            // x, y , w , h
            boxes.add(
                    new Rectangle(
                            array[2],
                            array[3],
                            (array[4] - array[2]),
                            (array[5] - array[3])));
        }
        return new DetectedObjects(names, prob, boxes);
    }
}
