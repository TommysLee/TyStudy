package com.ty.ai.cv.paddlepaddle.models.ppyoloe;

import ai.djl.modality.Output;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.ndarray.NDManager;
import com.google.common.collect.Lists;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import lombok.extern.slf4j.Slf4j;

import java.util.List;

/**
 * Paddle YoloE Plus Non-Standard Translator
 *
 * @Author Tommy
 * @Date 2024/5/3
 */
@Slf4j
public class PPYoloEPlusTranslator {

    private Gson gson = new Gson();

    /**
     * 数据后置处理
     *
     * @param manager   NDArray Manager
     * @param outputs   Python输出的的标准格式
     * @param threshold 阈值 - 目标检测成功的阈值
     * @return DetectedObjects the detected objects results for a single image
     * @throws Exception
     */
    public DetectedObjects processOutput(NDManager manager, Output outputs, Double threshold) throws Exception {
        if (null != outputs) {
            if (outputs.getCode() == 200) {
                String boxes = outputs.getAsString("boxes");
                String labels = outputs.getAsString("labels");
                String scores = outputs.getAsString("scores");

                // JSON反序列化
                List<List<Double>> boxList = gson.fromJson(boxes, new TypeToken<>() {});
                List<String> labelList = gson.fromJson(labels, new TypeToken<>() {});
                List<Double> scoreList = gson.fromJson(scores, new TypeToken<>() {});

                // 筛选出达到阈值的Item
                List<List<Double>> filteredBoxList = Lists.newArrayList();
                List<String> filteredLabelList = Lists.newArrayList();
                List<Double> filteredScoreList = Lists.newArrayList();
                for (int i = 0; i < scoreList.size(); i++) {
                    double score = scoreList.get(i);
                    if (score >= threshold) {
                        filteredBoxList.add(boxList.get(i));
                        filteredLabelList.add(labelList.get(i));
                        filteredScoreList.add(score);
                    }
                }

                // 将Box转换为BoundingBox
                List<BoundingBox> boundingBoxList = Lists.newArrayListWithCapacity(filteredBoxList.size());
                for (List<Double> box : filteredBoxList) {
                    double xmin = box.get(0);
                    double ymin = box.get(1);
                    double xmax = box.get(2);
                    double ymax = box.get(3);

                    double w = xmax - xmin;
                    double h = ymax - ymin;

                    boundingBoxList.add(new Rectangle(xmin, ymin, w, h));
                }

                // Build DetectedObjects
                return new DetectedObjects(filteredLabelList, filteredScoreList, boundingBoxList);
            } else {
                log.error("Python handle() failed: " + outputs.getMessage());
            }
        }
        return null;
    }
}
