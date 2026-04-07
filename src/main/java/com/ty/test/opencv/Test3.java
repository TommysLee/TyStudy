package com.ty.test.opencv;

import com.ty.utils.CVUtil;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.videoio.VideoCapture;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;

/**
 * OpenCV 读取视频 最后1帧
 *
 * @Author Tommy
 * @Date 2024/5/7
 */
public class Test3 extends JFrame {

    private JLabel label;

    public Test3() {
        super("My Video"); // 设置窗口标题

        // 创建JLabel并设置图片
        this.label = new JLabel();

        // 设置JFrame的布局（如果需要）
        setLayout(new BorderLayout());

        // 将JLabel添加到JFrame的内容窗格中
        getContentPane().add(label, BorderLayout.CENTER); // 使用BorderLayout并居中显示图片

        // 设置JFrame的大小
        // setSize(scaledIcon.getIconWidth() + 50, scaledIcon.getIconHeight() + 100); // 加上一些额外的空间以适应窗口边框等
        // 或者使用setPreferredSize()和pack()方法
        setSize(800, 600);

        // 设置JFrame的位置（如果需要）
        setLocationRelativeTo(null); // 居中显示窗口

        // 设置默认关闭操作
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        // 显示JFrame
        setVisible(true);
    }

    void setImage(BufferedImage image) {
        // 如果需要，可以调整图片大小以适应窗口
        Image scaledImage = image.getScaledInstance(image.getWidth(), image.getHeight(), Image.SCALE_SMOOTH);
        ImageIcon scaledIcon = new ImageIcon(scaledImage);
        this.label.setIcon(scaledIcon);
    }

    public static void main(String[] args) {
        // 加载 OpenCV 本地库（两种方法均可）
        // System.load(Test2.class.getClassLoader().getResource("opencv_java470.dll").getPath());
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        long begin = System.currentTimeMillis();

        // 打开视频文件/摄像头
        VideoCapture cap = new VideoCapture(0);
        System.out.println(cap);

        // 检查视频文件是否成功打开
        if (!cap.isOpened()) {
            System.out.println("Error opening video file.");
            return;
        }

        Test3 win = new Test3();

        // 读取最后1帧
        Mat last_frame = new Mat();
        while (true) {
            Mat frame = new Mat();
            boolean ret = cap.read(frame);
            if (ret) {
                last_frame = frame;
                BufferedImage bufferedImage = CVUtil.mat2Image(last_frame);
                win.setImage(bufferedImage);
            } else {
                break;
            }
        }

        if (!last_frame.empty()) {
            Imgcodecs.imwrite("v_frame_last.jpg", last_frame);
            System.out.println("Visualization file saved in ./v_frame_last.jpg");
        } else {
            System.out.println("Can't receive frame (stream end?).");
        }

        // 释放资源
        cap.release();
        last_frame.release();

        long end = System.currentTimeMillis();
        System.out.printf("耗时： %s ms%n", end - begin);
    }
}
