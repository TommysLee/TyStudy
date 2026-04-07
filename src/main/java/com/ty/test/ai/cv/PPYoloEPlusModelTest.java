package com.ty.test.ai.cv;

import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.DetectedObjects.DetectedObject;
import com.google.common.collect.Lists;
import com.ty.ai.cv.paddlepaddle.models.ppyoloe.PPYoloEPlusModel;
import com.ty.utils.CVUtil;
import com.ty.utils.PathRootUtil;
import lombok.extern.slf4j.Slf4j;

import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Paddle YoloE Plus Model Test
 *
 * @Author Tommy
 * @Date 2024/5/3
 */
@Slf4j
public class PPYoloEPlusModelTest {

    static String modelPath = "D:/opt/paddle_models/ppyoloE_plus";
    static Integer gpu = 0;

    static String image1 = "src/main/resources/test/test1.jpg";
    static String image2 = "src/main/resources/test/test2.jpg";

    static final String PROJ_ROOT;

    static {
        PROJ_ROOT = PathRootUtil.root();
    }

    /**
     * CPU 单线程测试
     *
     * @throws Exception
     */
    static void t1_cpu() throws Exception {
        String imgPath = PROJ_ROOT + image1;

        // 模型初始化
        PPYoloEPlusModel model = new PPYoloEPlusModel(modelPath).initialize();

        log.info("开始推理：{}", imgPath);

        DetectedObjects detectedObjects = model.predict(imgPath);

        log.info("推理结束：{}", detectedObjects);

        System.out.println("\r\n检测到的Person：");
        List<DetectedObject> personList = model.filter_person(detectedObjects);
        System.out.println(personList);

        CVUtil.showResult(imgPath, detectedObjects);

        // 释放资源
        model.close();
    }

    /**
     * GPU 单线程测试
     *
     * @throws Exception
     */
    static void t1_gpu() throws Exception {
        String imgPath = PROJ_ROOT + image1;

        // 模型初始化
        PPYoloEPlusModel model = new PPYoloEPlusModel(modelPath, gpu).initialize();

        log.info("开始推理：{}", imgPath);

        DetectedObjects detectedObjects = model.predict(imgPath);

        log.info("推理结束：{}", detectedObjects);

        System.out.println("\r\n检测到的Person：");
        List<DetectedObject> personList = model.filter_person(detectedObjects);
        System.out.println(personList);

        CVUtil.showResult(imgPath, detectedObjects);

        // 等价于 CVUtil.showResult
        // Image image = ImageFactory.getInstance().fromFile(Paths.get(imgPath));
        // image.drawBoundingBoxes(detectedObjects);
        // image.save(Files.newOutputStream(Paths.get(imgPath + "_result.png")), "png");

        // 释放资源
        model.close();
    }

    /**
     * 多线程测试
     *
     * @throws Exception
     */
    static void t2() throws Exception {
        // 模型初始化
        PPYoloEPlusModel model = new PPYoloEPlusModel(modelPath, gpu).initialize();

        // 先预测1次，让Python Model Load Complete
        model.predict(PROJ_ROOT + image1);

        // 开始多线程
        int nThreads = 5;
        ExecutorService executorService = Executors.newFixedThreadPool(nThreads);
        List<Callable<Long>> tasks = Lists.newArrayListWithCapacity(nThreads);
        for (int i = 0; i < nThreads; i++) {
            tasks.add(() -> {
                long threadId = Thread.currentThread().getId();
                String img = PROJ_ROOT + (threadId % 2 == 0? image1 : image2);

                log.info("开始推理：{}", img);

                DetectedObjects detectedObjects = model.predict(img);

                log.info("推理完成： {}", img);
                return threadId;
            });
        }

        long begin = System.currentTimeMillis();
        List<Future<Long>> futures = executorService.invokeAll(tasks);
        for (Future<Long> f : futures) {
            log.info("Thread-{} Completed", f.get());
        }
        long end = System.currentTimeMillis();

        log.info("All Completed! 总耗时：{}ms", (end - begin));

        // 释放资源
        model.close();
    }

    /*
     * 测试
     */
    public static void main(String[] args) throws Exception {
        t1_cpu();
        // t1_gpu();

        // t2();

        System.exit(0);
    }
}
