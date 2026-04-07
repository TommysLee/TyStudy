package com.ty.test.boofcv;

import boofcv.abst.tracker.TrackerObjectQuad;
import boofcv.factory.tracker.FactoryTrackerObjectQuad;
import boofcv.gui.image.ShowImages;
import boofcv.gui.tracker.TrackerObjectQuadPanel;
import boofcv.io.image.ConvertBufferedImage;
import boofcv.misc.BoofMiscOps;
import boofcv.struct.image.GrayU8;
import georegression.struct.shapes.Quadrilateral_F64;
import lombok.extern.slf4j.Slf4j;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

import java.awt.*;
import java.awt.image.BufferedImage;

/**
 * BoofCV 目标追踪示例
 *
 * @Author Tommy
 * @Date 2024/5/9
 */
@Slf4j
public class Test1 {

    static Quadrilateral_F64 location;

    static {
        // 加载 OpenCV 本地库
        nu.pattern.OpenCV.loadShared();
    }

    /**
     * 创建目标追踪器
     */
    static TrackerObjectQuad<GrayU8> createTracker(GrayU8 frame) {
        // Create the tracker
        TrackerObjectQuad<GrayU8> tracker = FactoryTrackerObjectQuad.circulant(null, GrayU8.class);

        // 指定目标的初始位置
        // 695.022095,282.770325, 899.071899, 610.615601
        // 1153.963501,21.406242, 1286.153076, 380.996704
        // 1083.330322,-1.754654, 1183.339111, 281.343933
        // 195 302 716 1122
        double xmin = 695;
        double ymin = 282;
        double xmax = 900;
        double ymax = 611;

        // 左上角：x0、y0
        double x0 = xmin;
        double y0 = ymin;

        // 右上角：x1、y1
        double x1 = xmax;
        double y1 = ymin;

        // 右下角：x2、y2
        double x2 = xmax;
        double y2 = ymax;

        // 左下角：x3、y3
        double x3 = xmin;
        double y3 = ymax;

        location = new Quadrilateral_F64(x0, y0, x1, y1, x2, y2, x3, y3);

        // 初始化目标位置
        boolean flag = tracker.initialize(frame, location);

        log.info("目标定位{}", flag? "成功" : "失败");

        return tracker;
    }

    /**
     * OpenCV Mat 转 BufferedImage
     */
    static BufferedImage mat2Image(Mat mat) {
        int width = mat.width();
        int height = mat.height();
        byte[] data = new byte[width * height * (int) mat.elemSize()];
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_BGR2RGB);
        mat.get(0, 0, data);
        BufferedImage ret = new BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR);
        ret.getRaster().setDataElements(0, 0, width, height, data);
        return ret;
    }

    /*
     * main
     */
    public static void main(String[] args) {
        // 打开视频文件
        String vpath = "C:/DevTools/MediaMTX/recordings/live/2024-05-09_12-10-19-303746.mp4";
        VideoCapture cap = new VideoCapture(vpath);

        // 检查视频文件是否成功打开
        if (!cap.isOpened()) {
            log.info("Error opening video file.");
            return;
        }
        log.info("视频流加载成功 {}", vpath);

        // 获取第1帧图像
        Mat mat = new Mat();
        boolean ret = cap.read(mat);
        BufferedImage guiImage = mat2Image(mat);
        GrayU8 frame = ConvertBufferedImage.convertFrom(guiImage, (GrayU8) null);

        // 初始化目标追踪器
        TrackerObjectQuad<GrayU8> tracker = createTracker(frame);

        // For displaying the results
        TrackerObjectQuadPanel gui = new TrackerObjectQuadPanel(null);
        gui.setPreferredSize(new Dimension(frame.getWidth(), frame.getHeight()));
        gui.setImageUI(guiImage);
        gui.setTarget(location, true);
        ShowImages.showWindow(gui, "Tracking Results", true);

        // Track the object across each video frame and display the results
        long previous = 0;
        boolean loss = false;
        while (true) {
            ret = cap.read(mat);
            if (ret) {
                guiImage = mat2Image(mat);
                frame = ConvertBufferedImage.convertFrom(guiImage, (GrayU8) null);

                boolean visible = tracker.process(frame, location);
                gui.setImageUI(guiImage);
                gui.setTarget(location, visible);
                gui.repaint();

                if (!visible && !loss) {
                    loss = true;
                    log.info("目标消失");
                }

                // shoot for a specific frame rate
                long time = System.currentTimeMillis();
                BoofMiscOps.pause(Math.max(0, 80 - (time - previous)));
                previous = time;
            } else {
                break;
            }
        }

        // 释放资源
        cap.release();
        mat.release();
        log.info("释放资源");
    }
}
