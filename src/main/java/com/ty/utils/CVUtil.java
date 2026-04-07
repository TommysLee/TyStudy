package com.ty.utils;

import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.DetectedObjects.DetectedObject;
import ai.djl.modality.cv.output.Rectangle;
import georegression.struct.shapes.Rectangle2D_I32;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.codec.binary.Base64;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.geom.Rectangle2D;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.util.List;

/**
 * Computer Vision Util
 *
 * @Author Tommy
 * @Date 2024/5/3
 */
@Slf4j
public class CVUtil {

    /**
     * 绘制图形
     *
     * @param image                 BufferedImage
     * @param detectedObjectList    DetectedObject List
     * @return BufferedImage
     */
    public static BufferedImage draw(BufferedImage image, List<DetectedObject> detectedObjectList) {
        Font font = new Font("微软雅黑", Font.PLAIN, 14);
        Graphics2D g = (Graphics2D) image.getGraphics();
        g.setFont(font);
        g.setStroke(new BasicStroke(2.0f));
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON); // 抗锯齿
        FontMetrics fm = g.getFontMetrics();
        AlphaComposite alphaChannel = AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 0.65f);

        for (DetectedObject item : detectedObjectList) {
            String label = item.getClassName();
            String text = label + " " + (Math.round(item.getProbability() * 100d) / 100d);
            Rectangle2D labelRect = fm.getStringBounds(text, g);

            Rectangle rect = item.getBoundingBox().getBounds();
            int x = (int) Math.round(rect.getX());
            int y = (int) Math.round(rect.getY());

            g.setColor(ColorUtil.color(label));
            g.drawRect(x, y, (int) Math.round(rect.getWidth()), (int) Math.round(rect.getHeight()));

            g.setComposite(alphaChannel);
            g.fillRect(x, y, (int) (labelRect.getWidth() + 5), (int) (labelRect.getHeight() + 5));
            g.setComposite(AlphaComposite.SrcOver);

            y = (int) (y + labelRect.getHeight());
            g.setColor(Color.WHITE);
            g.drawString(text, x, y);
        }

        g.dispose();
        return image;
    }

    /**
     * Show DetectedObject Result
     *
     * @param imagePath             Image File Path
     * @param detectedObjectList    DetectedObject List
     * @return String 返回绘制后的图片路径
     */
    public static String showResult(String imagePath, List<DetectedObject> detectedObjectList) throws IOException {
        File imgFile = new File(imagePath);
        BufferedImage image = ImageIO.read(imgFile);
        draw(image, detectedObjectList);

        String fileFullName = imgFile.getName();
        int pointIndex = fileFullName.lastIndexOf(".");
        String fileName = fileFullName.substring(0, pointIndex);
        String suffix = fileFullName.substring(pointIndex);

        String destPath = imgFile.getParent() + File.separator + fileName + "_visualized_result" + suffix;
        File destFile = new File(destPath);
        if (!destFile.getParentFile().exists()) {
            destFile.mkdirs();
        }
        ImageIO.write(image, suffix.substring(1), new File(destPath));

        log.info("可视化 Visualized result save in {}", destPath);

        return destPath;
    }

    /**
     * Show DetectedObject Result
     *
     * @param imagePath         Image File Path
     * @param detectedObjects   DetectedObjects
     * @return String 返回绘制后的图片路径
     */
    public static String showResult(String imagePath, DetectedObjects detectedObjects) throws IOException {
        return showResult(imagePath, detectedObjects.items());
    }

    /**
     * DJL Rectangle 转 BoofCV Rectangle2D_I32
     *
     * @param rect  DJL Rectangle
     * @param scale Scale
     * @return BoofCV Rectangle2D_I32
     */
    public static Rectangle2D_I32 rectangle2Rectangle2D(Rectangle rect, double scale) {
        double minX = rect.getX();
        double minY = rect.getY();
        double maxX = minX + rect.getWidth();
        double maxY = minY + rect.getHeight();
        return new Rectangle2D_I32((int) (minX * scale), (int) (minY * scale), (int) (maxX * scale), (int) (maxY * scale));
    }

    /**
     * BufferedImage 转 Base64
     */
    public static String image2Base64(BufferedImage image) {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        try {
            ImageIO.write(image, "JPEG", baos);
        } catch (Exception e) {
            log.error(e.getMessage(), e);
        }
        return Base64.encodeBase64String(baos.toByteArray());
    }

    /**
     * OpenCV Mat 转 BufferedImage
     */
    public static BufferedImage mat2Image(Mat mat) {
        int width = mat.width();
        int height = mat.height();
        byte[] data = new byte[width * height * (int) mat.elemSize()];
        Imgproc.cvtColor(mat, mat, 4);
        mat.get(0, 0, data);
        BufferedImage ret = new BufferedImage(width, height, 5);
        ret.getRaster().setDataElements(0, 0, width, height, data);
        return ret;
    }
}
