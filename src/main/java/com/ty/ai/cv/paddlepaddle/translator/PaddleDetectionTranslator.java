package com.ty.ai.cv.paddlepaddle.translator;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.Image.Interpolation;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.TranslatorContext;
import com.google.common.collect.Lists;
import com.ty.ai.cv.paddlepaddle.Preprocess;
import lombok.extern.slf4j.Slf4j;
import org.yaml.snakeyaml.Yaml;
import tech.tablesaw.api.FloatColumn;
import tech.tablesaw.api.Row;
import tech.tablesaw.api.Table;

import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Paddle目标检测Translator
 *
 * @Author Tommy
 * @Date 2024/5/14
 */
@Slf4j
public class PaddleDetectionTranslator implements NoBatchifyTranslator<Image, DetectedObjects> {

    protected int[] target_size;
    protected boolean keep_ratio;
    protected Interpolation interp;
    protected float threshold;
    protected int width;
    protected int height;
    protected List<String> labels;

    public PaddleDetectionTranslator(int target_width, int target_height, boolean keep_ratio) {
        this(target_width, target_height, keep_ratio, 0.5f);
    }

    public PaddleDetectionTranslator(int target_width, int target_height, boolean keep_ratio, float threshold) {
        this(target_width, target_height, keep_ratio, threshold, Interpolation.BILINEAR);
    }

    public PaddleDetectionTranslator(int target_width, int target_height, boolean keep_ratio, float threshold, Interpolation interp) {
        this.target_size = new int[] {target_height, target_width};
        this.keep_ratio = keep_ratio;
        this.threshold = threshold;
        this.interp = interp;
    }

    /**
     * Processes the input and converts it to NDList.
     *
     * @param ctx   the toolkit for creating the input NDArray
     * @param input the input object
     * @return the {@link NDList} after pre-processing
     * @throws Exception if an error occurs during processing input
     */
    @Override
    public NDList processInput(TranslatorContext ctx, Image input) throws Exception {
        log.debug("Process Input 数据预处理...");
        long begin = System.currentTimeMillis();

        this.width = input.getWidth();
        this.height = input.getHeight();
        NDArray im = input.toNDArray(ctx.getNDManager());

        // Resize
        List<NDArray> im_results = Preprocess.resize(im, this.target_size, this.keep_ratio, this.interp, ctx.getNDManager());
        im = im_results.get(0);
        NDArray scale_factor = im_results.get(1);

        // NormalizeImage
        im = Preprocess.normalizeImage(im, ctx.getNDManager());

        // Permute
        im = Preprocess.permute(im, ctx.getNDManager());

        long end = System.currentTimeMillis();
        log.debug("Process Input 数据预处理完成，耗时：{}ms", end - begin);

        return new NDList(im, scale_factor);
    }

    /**
     * Processes the output NDList to the corresponding output object.
     *
     * @param ctx  the toolkit used for post-processing
     * @param list the output NDList after inference, usually immutable in engines like
     *             PyTorch. @see <a href="https://github.com/deepjavalibrary/djl/issues/1774">Issue 1774</a>
     * @return the output object of expected type
     * @throws Exception if an error occurs during processing output
     */
    @Override
    public DetectedObjects processOutput(TranslatorContext ctx, NDList list) throws Exception {
        log.debug("Process Output 数据后处理...");
        long begin = System.currentTimeMillis();

        NDArray result = list.get(0);
        float[] probabilities = result.get(":,1").toFloatArray();
        List<String> names = new ArrayList<>();
        List<Double> probs = new ArrayList<>();
        List<BoundingBox> boxes = new ArrayList<>();
        for (int i = 0; i < probabilities.length; i++) {
            if (probabilities[i] >= this.threshold) {
                float[] item = result.get(i).toFloatArray();
                names.add(labels.get((int)item[0]));
                probs.add((double) probabilities[i]);
                boxes.add(
                        new Rectangle(
                                item[2],
                                item[3],
                                item[4] - item[2],
                                item[5] - item[3]));
            }
        }

        long end = System.currentTimeMillis();
        log.debug("Process Output 数据后处理完成，耗时：{}ms", end - begin);
        if (log.isTraceEnabled()) {
            printResult(result);
        }
        return new DetectedObjects(names, probs, boxes);
    }

    /**
     * 设置标签集
     *
     * @param labels Labels
     */
    public void setLabels(List<String> labels) {
        this.labels = labels;
    }

    /**
     * 加载标签集
     *
     * @param cfgPath Model Yaml Configuration File Path
     * @return <T extends PaddleDetectionTranslator>
     */
    @SuppressWarnings("unchecked")
    public <T extends PaddleDetectionTranslator> T loadLabels(String cfgPath) {
        List<String> label_list = Lists.newArrayList();
        try {
            Yaml yaml = new Yaml();
            Map<String, Object> data = yaml.load(new FileReader(cfgPath));
            label_list = (List<String>) data.get("label_list");
        } catch (Exception e) {
            log.error(e.getMessage(), e);
        } finally {
            this.setLabels(label_list);
        }
        return (T) this;
    }

    /**
     * 打印数据后处理的结果集
     */
    protected void printResult(NDArray result) {
        // 数据集格式如下：
        //     label  |         score          |         x_min         |        y_min         |        x_max         |        y_max         |
        //    -------------------------------------------------------------------------------------------------------------------------------
        //        33  |    0.9723130464553833  |   161.77688598632812  |   81.42154693603516  |    616.650146484375  |   344.8529052734375  |
        //         0  |    0.9130071401596069  |    163.8280487060547  |   81.87583923339844  |   198.5448455810547  |   167.5631103515625  |
        //         0  |      0.90949547290802  |   267.20965576171875  |    89.6672592163086  |    298.622314453125  |  169.03573608398438  |
        //         0  |    0.9025657176971436  |    581.2205200195312  |  113.08621978759766  |    612.760498046875  |  198.45480346679688  |
        //         0  |    0.8894960880279541  |   414.94024658203125  |    89.5226058959961  |   506.9897155761719  |     285.47021484375  |
        //         0  |    0.8093008995056152  |   503.78594970703125  |  113.48980712890625  |   594.9880981445312  |   271.4203796386719  |
        //         0  |    0.8059836626052856  |   364.07379150390625  |  58.571292877197266  |   381.6502685546875  |  114.45134735107422  |

        Table table = Table.create("目标检测结果集",
                FloatColumn.create("label"), FloatColumn.create("score"),
                FloatColumn.create("x_min"), FloatColumn.create("y_min"),
                FloatColumn.create("x_max"), FloatColumn.create("y_max"));

        float[] probabilities = result.get(":,1").toFloatArray();
        for (int i = 0; i < probabilities.length; i++) {
            Row row = table.appendRow();
            float[] item = result.get(i).toFloatArray();
            row.setFloat(0, item[0]);
            row.setFloat(1, item[1]);
            row.setFloat(2, item[2]);
            row.setFloat(3, item[3]);
            row.setFloat(4, item[4]);
            row.setFloat(5, item[5]);
        }
        table.setName(table.name() + "（" + table.rowCount() + "条）");
        log.trace("\r\n{}\r\n", table);
        table = null;
    }
}
