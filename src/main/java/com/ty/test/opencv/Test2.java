package com.ty.test.opencv;

import lombok.extern.slf4j.Slf4j;
import nu.pattern.OpenCV;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.videoio.VideoCapture;
import org.slf4j.bridge.SLF4JBridgeHandler;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * OpenCV 读取视频 第1帧
 *
 * @Author Tommy
 * @Date 2024/5/7
 */
@Slf4j
public class Test2 {

    static {
        SLF4JBridgeHandler.install();
        Logger globalLogger = Logger.getLogger(OpenCV.class.getName());
        globalLogger.setLevel(Level.FINEST);

        // 加载 OpenCV 本地库（两种方法均可）
        // System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        // nu.pattern.OpenCV.loadLocally(); JDK12+
        nu.pattern.OpenCV.loadShared();
    }

    static void cap(String prefix, int index) {
        long begin = System.currentTimeMillis();

        String vpath = "Amaze" + index + ".mp4";

        log.info("开始Start....{}", vpath);

        if (index % 2 != 0) {
            try {
                Thread.currentThread().sleep(2000);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        // 打开视频文件/摄像头
        VideoCapture cap = new VideoCapture();
        cap.open(vpath);

        // 检查视频文件是否成功打开
        if (!cap.isOpened()) {
            System.out.println("Error opening video file." + vpath);
            return;
        }

        // 读取第一帧
        // cap.set(Videoio.CAP_PROP_POS_FRAMES, 5);
        Mat frame = new Mat();
        boolean ret = cap.read(frame);

        long end = System.currentTimeMillis();
        log.info("耗时： {}ms", end - begin);

        // Write Image to Disk
        if (ret) {
            Imgcodecs.imwrite(prefix + "v_frame_1.jpg", frame);
            log.info("Visualization file saved in ./{}v_frame_1.jpg", prefix);
        } else {
            log.info("Can't receive frame (stream end?).");
        }

        // 释放资源
        cap.release();
        frame.release();
    }

    /*
     * 多线程测试
     */
    public static void main(String[] args) throws Exception {
        int nThreads = 1;
        ExecutorService executorService = Executors.newFixedThreadPool(nThreads);
        Collection<Callable<String>> tasks = new ArrayList<>(nThreads);

        AtomicInteger n = new AtomicInteger();
        for (int i = 0; i < nThreads; i++) {
            tasks.add(() -> {
                cap(Thread.currentThread().getName() + "_", n.getAndIncrement());
                return Thread.currentThread().getName() + " SUCCESS";
            });
        }

        List<Future<String>> results = executorService.invokeAll(tasks);
        for (Future<String> r : results) {
            System.out.println(r.get());
        }
        System.out.println("程序结束");
        System.exit(0);
    }
}
