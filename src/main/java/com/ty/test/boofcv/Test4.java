package com.ty.test.boofcv;

import boofcv.alg.distort.RemovePerspectiveDistortion;
import boofcv.gui.image.ShowImages;
import boofcv.io.image.ConvertBufferedImage;
import boofcv.io.image.UtilImageIO;
import boofcv.struct.image.GrayF32;
import boofcv.struct.image.ImageType;
import boofcv.struct.image.Planar;
import com.ty.utils.PathRootUtil;
import georegression.struct.point.Point2D_F64;
import lombok.extern.slf4j.Slf4j;

import java.awt.image.BufferedImage;

/**
 * 仿射变换 Test Case
 *
 * https://zhuanlan.zhihu.com/p/465490024
 *
 * @Author Tommy
 * @Date 2024/5/9
 */
@Slf4j
public class Test4 {

    static String fileName = "src/main/resources/test/goals_and_stuff.jpg";
    // static String fileName = "src/main/resources/test/goals_paper.jpg";
    static double[][] region = new double[][] {
            {267, 182},     // 左上角
            {542, 68},      // 右上角
            {519, 736},     // 右下角
            {276, 570}      // 左下角

            /* {360, 595},     // 左上角
            {1464, 267},    // 右上角
            {1525, 2432},   // 右下角
            {315, 2339}     // 左下角 */
    };

    public static void main(String[] args) {
        // 加载图像
        String imagePath = PathRootUtil.resolve(fileName).toString();
        BufferedImage buffered = UtilImageIO.loadImageNotNull(imagePath);
        Planar<GrayF32> input = ConvertBufferedImage.convertFromPlanar(buffered, null, true, GrayF32.class);
        log.info("图像加载成功 {}x{} {}", buffered.getWidth(), buffered.getHeight(), imagePath);

        // 仿射算法对象
        RemovePerspectiveDistortion<Planar<GrayF32>> removePerspective =
                new RemovePerspectiveDistortion<>(buffered.getWidth(), buffered.getHeight(), ImageType.pl(3, GrayF32.class));

        // Specify the corners in the input image of the region.
        // 四个角的顺序：左上、右上、右下、左下
        if (!removePerspective.apply(input,
                new Point2D_F64(region[0][0], region[0][1]), new Point2D_F64(region[1][0], region[1][1]),
                new Point2D_F64(region[2][0], region[2][1]), new Point2D_F64(region[3][0], region[3][1]))) {
            throw new RuntimeException("仿射变换失败!!!");
        }

        Planar<GrayF32> output = removePerspective.getOutput();
        BufferedImage flat = ConvertBufferedImage.convertTo_F32(output, null, true);
        log.info("图像矫正成功 {}x{} {}", flat.getWidth(), flat.getHeight(), imagePath);

        ShowImages.showWindow(buffered, "原图片", true);
        ShowImages.showWindow(flat, "矫正图片", true);
    }
}
