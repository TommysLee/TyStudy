package com.ty.test.boofcv;

import boofcv.io.MediaManager;
import boofcv.io.image.SimpleImageSequence;
import boofcv.io.wrapper.DefaultMediaManager;
import boofcv.struct.image.GrayU8;
import boofcv.struct.image.ImageType;

/**
 * BoofCV 读取 MPEG 视频示例
 *
 * @Author Tommy
 * @Date 2024/5/6
 */
public class Test5 {

    public static void main(String[] args) throws Exception {
        String fileName = "tree.mjpeg";

        MediaManager media = DefaultMediaManager.INSTANCE;

        long begin = System.currentTimeMillis();

        SimpleImageSequence<GrayU8> sequence = media.openVideo(fileName, ImageType.single(GrayU8.class));

        long middle = System.currentTimeMillis();

        System.out.println(String.format("视频加载完毕 耗时：%s ms", middle - begin));
        System.out.println(sequence);

        // int tick = 1;
        // int fps = 0;
        // while (sequence.hasNext()) {
        //     sequence.next();
        //     if (fps % 10 == 0) {
        //         BufferedImage bufferedImage = sequence.getGuiImage();
        //         ImageIO.write(bufferedImage, "JPEG", new File("D:/tt/" + (tick++) + ".jpg"));
        //     }
        //     fps++;
        // }

        long end = System.currentTimeMillis();

        System.out.println(String.format("视频转图片 耗时：%s ms", end - middle));

        System.out.println(String.format("程序完成 耗时：%s ms", end - begin));
    }
}
