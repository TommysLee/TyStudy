package com.ty.test.ai.cv;

import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.DetectedObjects.DetectedObject;
import ai.djl.modality.cv.output.Rectangle;
import boofcv.gui.BoofSwingUtil;
import boofcv.gui.feature.VisualizeShapes;
import boofcv.gui.fiducial.VisualizeFiducial;
import boofcv.gui.image.ImagePanel;
import boofcv.gui.image.ScaleOptions;
import com.arcsoft.face.Rect;
import com.google.common.collect.Lists;
import com.ty.ai.cv.face.Face;
import com.ty.ai.cv.face.FaceRecognitionResult;
import com.ty.ai.cv.paddlepaddle.models.ppyoloe.PPYoloEPlusONNXModel;
import com.ty.utils.CVUtil;
import com.ty.utils.ColorUtil;
import com.ty.utils.ThreadUtil;
import georegression.struct.point.Point2D_F64;
import georegression.struct.shapes.Rectangle2D_I32;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.apache.commons.lang3.math.NumberUtils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.videoio.VideoCapture;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;
import java.util.List;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.locks.ReentrantLock;

/**
 * 视频流实时AI检测 Ultra版（目标检测 + 人脸识别）based on DJL ONNX Engine
 *
 * @Author Tommy
 * @Date 2024/5/15
 */
@Slf4j
public class RTLiveDetectedObjectsOnnXUltraTest extends ImagePanel {

    private JFrame jFrame;
    private final int fontSize = 12;
    private final Font font = new Font("Serif", Font.BOLD, fontSize);
    private final Color black130 = new Color(0, 0, 0, 130);
    private DetectedObjects detectedObjects;

    static String VPATH = "rtsp://admin:easygrid.wh@172.16.3.24:554/Streaming/Channels/101";
    static VideoCapture cap;
    static Mat frame;

    static PPYoloEPlusONNXModel MODEL;
    static String MODEL_PATH = "/opt/onnx_models/ppyoloE_plus";

    static Face face;

    static final LinkedBlockingQueue<BufferedImage> QUEUE = new LinkedBlockingQueue<>(1);
    static ReentrantLock LOCK = new ReentrantLock();

    public RTLiveDetectedObjectsOnnXUltraTest() {
        super.setScaling(ScaleOptions.DOWN);
    }

    public RTLiveDetectedObjectsOnnXUltraTest(BufferedImage image) {
        super(image, ScaleOptions.DOWN);
    }

    static {
        // 读取配置信息
        VPATH = StringUtils.defaultIfBlank(System.getProperty("url"), VPATH);
        MODEL_PATH = StringUtils.defaultIfBlank(System.getProperty("model"), MODEL_PATH);

        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    /**
     * GUI初始化
     */
    int a = 0;
    public RTLiveDetectedObjectsOnnXUltraTest init() {
        setBackground(Color.BLACK);
        addMouseListener(new MouseAdapter() { // 添加视频双击事件：全屏/退出全屏
            @Override
            public void mouseClicked(MouseEvent e) {
                if (e.getClickCount() % 2 == 0) {
                    GraphicsDevice gd = GraphicsEnvironment.getLocalGraphicsEnvironment().getDefaultScreenDevice();
                    if (gd.isFullScreenSupported()) {
                        if (null == gd.getFullScreenWindow()) { // 全屏
                            gd.setFullScreenWindow(jFrame);
                        } else { // 退出全屏
                            gd.setFullScreenWindow(null);
                        }
                    }
                }
            }
        });

        jFrame = new JFrame("RTLive AI实时感知");
        jFrame.setSize(1024, 768);
        jFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        jFrame.setLocationByPlatform(true);
        jFrame.add(this, BorderLayout.CENTER);
        jFrame.setVisible(true);
        jFrame.addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) { // 添加窗体关闭事件
                // 窗体关闭时，释放资源
                try {
                    LOCK.lock();
                    cap.release();
                    frame.release();
                    if (null != MODEL) {
                        MODEL.close();
                    }
                } catch (Exception ex) {
                    log.error(ex.getMessage(), ex);
                }
                log.info("\r\nGUI窗体关闭，资源已释放\r\n");
            }
        });

        if (getImage().getWidth() < 700) {
            jFrame.pack();
        } else {
            jFrame.setExtendedState(JFrame.MAXIMIZED_BOTH);
        }

        log.info("GUI窗体初始化完成");
        return this;
    }

    /**
     * 绘制GUI组件
     */
    @Override
    public void paintComponent(Graphics g) {
        super.paintComponent(g);
        log.debug("GUI绘制 width={} height={} scale={} offsetX={} offsetY={}", getWidth(), getHeight(), this.scale, this.offsetX, this.offsetY);

        // 绘制标记
        if (null != this.detectedObjects) {
            Graphics2D g2 = BoofSwingUtil.antialiasing(g);
            g2.setStroke(new BasicStroke(2));

            List<DetectedObject> items = this.detectedObjects.items();
            for (DetectedObject item : items) {
                String label = item.getClassName();
                // 矩形
                g2.setColor(ColorUtil.color(label));
                Rectangle rect = item.getBoundingBox().getBounds();
                Rectangle2D_I32 rect_i32 = CVUtil.rectangle2Rectangle2D(rect, this.scale);
                VisualizeShapes.drawRectangle(rect_i32, g2);

                // Label
                Point2D_F64 point = new Point2D_F64(rect_i32.getX0() + fontSize + 5, rect_i32.getY0() + fontSize);
                VisualizeFiducial.drawLabel(point, label, font, Color.ORANGE, black130, g2, 1);
            }
        }
    }

    /**
     * 更新帧图片
     */
    public void tick(BufferedImage image) {
        super.setImage(image);
        super.repaint();
    }

    /**
     * 发送数据给识别任务
     */
    public void send(BufferedImage image) {
        QUEUE.poll();
        QUEUE.offer(image);
    }

    /**
     * 是否在Jar包中运行
     */
    boolean isInJar() {
        String cpath = RTLiveDetectedObjectsOnnXUltraTest.class.getProtectionDomain().getCodeSource().getLocation().getPath();
        return cpath.contains(".jar");
    }

    /**
     * 启动异步AI识别任务
     */
    public void startAsyncTask() {
        try {
            new Thread(() -> {
                // 模型初始化
                try {
                    MODEL = new PPYoloEPlusONNXModel(MODEL_PATH).initialize();
                    face = new Face();
                } catch (Exception e) {
                    log.error(e.getMessage(), e);
                    System.exit(0);
                }

                List<FaceRecognitionResult> faceList = Lists.newArrayList();
                while (true) {
                    try {
                        BufferedImage image = QUEUE.take();
                        LOCK.lock();
                        DetectedObjects _detectedObjects = MODEL.predict(image);

                        if (_detectedObjects.getClassNames().contains("人")) {
                            long begin = System.currentTimeMillis();
                            faceList = face.detect(image);
                            // 目标名称替换为人员身份信息
                            if (!faceList.isEmpty()) {
                                List<DetectedObject> items = _detectedObjects.items();
                                int index = 0;
                                for (DetectedObject item : items) {
                                    if ("人".equals(item.getClassName())) {
                                        Rectangle rect = item.getBoundingBox().getBounds();
                                        for (FaceRecognitionResult fr : faceList) {
                                            Rect faceRect = fr.getFaceInfo().getRect();
                                            double x0 = rect.getX();
                                            double y0 = rect.getY();
                                            double x1 = rect.getWidth() + x0;
                                            double y1 = rect.getHeight() + y0;

                                            if (faceRect.getLeft() >= x0 && faceRect.getTop() >= y0 && faceRect.getRight() <= x1 && faceRect.getBottom() <= y1) {
                                                _detectedObjects.getClassNames().set(index, fr.getName());
                                            }
                                        }
                                    }
                                    index++;
                                }
                                long end = System.currentTimeMillis();
                                log.info("Face新增耗时：{}", end - begin);
                            }
                        }

                        this.detectedObjects = _detectedObjects;
                    } catch (Exception e) {
                        log.error(e.getMessage(), e);
                    } finally {
                        LOCK.unlock();
                        ThreadUtil.sleep(5);
                    }
                }
            }).start();
            log.info("AI识别任务已启动");
        } catch (Exception e) {
            log.error(e.getMessage(), e);
        }
    }

    /**
     * 加载视频流
     */
    static VideoCapture loadVideo() {
        log.info("视频流加载中... {}", VPATH);

        frame = new Mat();
        if (NumberUtils.isCreatable(VPATH)) {
            cap = new VideoCapture(Integer.parseInt(VPATH));
        } else {
            cap = new VideoCapture(VPATH);
        }

        if (!cap.isOpened()) {
            log.info("Error opening video stream.");
            System.exit(0);
        }
        cap.read(frame);
        log.info("视频流加载成功 {}", VPATH);
        return cap;
    }

    /*
     * 测试
     */
    public static void main(String[] args) throws Exception {
        // 打开视频流 并加载第1帧图像
        loadVideo();
        BufferedImage image = CVUtil.mat2Image(frame);

        // 创建GUI窗体
        RTLiveDetectedObjectsOnnXUltraTest live = new RTLiveDetectedObjectsOnnXUltraTest(image).init();
        live.startAsyncTask();
        while (true) {
            boolean ret = cap.read(frame);
            if (ret) {
                image = CVUtil.mat2Image(frame);
                live.tick(image);
                live.send(image);
                ThreadUtil.sleep(5);
            } else {
                break;
            }
        }
    }
}
