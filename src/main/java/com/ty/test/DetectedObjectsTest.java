package com.ty.test;

import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.DetectedObjects.DetectedObject;
import ai.djl.modality.cv.output.Rectangle;
import com.google.common.collect.Lists;

import java.util.List;

/**
 * 测试：手动构建 DetectedObjects 对象
 *
 * @Author Tommy
 * @Date 2024/5/2
 */
public class DetectedObjectsTest {

    static List<BoundingBox> boundingBoxes() {
        List<BoundingBox> boundingBoxes = Lists.newArrayList();
        int size = 10;
        for (int i = 0; i < 6; i++) {
            Rectangle rect = new Rectangle(i * size, i * size,  size, size);
            boundingBoxes.add(rect);
        }
        return boundingBoxes;
    }

    public static void main(String[] args) {
        List<String> classNames = Lists.newArrayList("Person", "Person", "Car", "Bike", "Bus", "Bird");
        List<Double> probabilities = Lists.newArrayList(0.8d, 0.72d, 0.8d, 0.4d, 0.5d, 0.41d);
        List<BoundingBox> boundingBoxes = boundingBoxes();

        DetectedObjects detectedObjects = new DetectedObjects(classNames, probabilities, boundingBoxes);
        detectedObjects.setTopK(5);

        System.out.println(detectedObjects);
        System.out.println(detectedObjects.getNumberOfObjects());
        System.out.println(detectedObjects.topK());
        System.out.println(detectedObjects.best());
        System.out.println();

        List<DetectedObject> personObjectList = Lists.newArrayList();
        for (int i = 0; i < detectedObjects.getNumberOfObjects(); i++) {
            DetectedObject dobj = detectedObjects.item(i);
            System.out.println(dobj);

            if ("Person".equalsIgnoreCase(dobj.getClassName())) {
                personObjectList.add(dobj);
            }
        }
        System.out.println();

        System.out.println("检测的人员如下：");
        System.out.println(personObjectList);
    }
}
