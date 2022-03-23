import sys
sys.path.append('../')
from pycore.tikzeng import *  # isort:skip # noqa
# defined your arch
arch = [
    to_head('..'),
    to_cor(),
    to_begin(),

    to_input('input', name='InputData'),

    to_Conv("conv1", 3, 32, offset="(0,0,0)",
            to="(0,0,0)", height=50, depth=50, width=2),
    to_BN("bn1", offset="(0,0,0)", to="(conv1-east)",
          height=25, depth=25, width=2, caption="BatchNorm1"),
    to_ReLU("relu1", offset="(0,0,0)", to="(bn1-east)",
            height=25, depth=25, width=2),


    to_Conv("conv2", 3, 64, offset="(2.5,0,0)",
            to="(relu1-east)", height=45, depth=45, width=2.5),
    to_BN("bn2", offset="(0,0,0)", to="(conv2-east)",
          height=25, depth=25, width=2, caption="BatchNorm2"),
    to_ReLU("relu2", offset="(0,0,0)", to="(bn2-east)",
            height=25, depth=25, width=2),

    to_connection("relu1", "conv2"),

    # 128 Size Block
    to_ReLU("relu3", offset="(2,0,0)", to="(relu2-east)",
            height=25, depth=25, width=2),
    to_Conv("conv3", 3, 128, offset="(0,0,0)",
            to="(relu3-east)", height=40, depth=40, width=3),
    to_BN("bn3", offset="(0,0,0)", to="(conv3-east)",
          height=25, depth=25, width=2, caption="BatchNorm3"),

    to_connection("relu2", "relu3"),

    to_ReLU("relu4", offset="(2,0,0)", to="(bn3-east)",
            height=25, depth=25, width=2),
    to_Conv("conv4", 3, 128, offset="(0,0,0)",
            to="(relu4-east)", height=40, depth=40, width=3),
    to_BN("bn4", offset="(0,0,0)", to="(conv4-east)",
          height=25, depth=25, width=2, caption="BatchNorm4"),

    to_connection("bn3", "relu4"),

    to_Pool("pool1", offset="(1.5,0,0)", to="(bn4-east)", height=40,
            depth=30, width=1, caption="MaxPool1"),  # 2x1
    # to_connection("bn1", "pool1"),

    to_connection("bn4", "pool1"),

    to_Sum("sum1", offset="(1,0,0)", to="(pool1-east)", radius=2, opacity=0.6),

    to_connection("pool1", "sum1"),
    to_skip("relu2", "sum1", pos_of=1.5, pos_to=7),

    # 256 Size Block

    to_ReLU("relu5", offset="(1,0,0)", to="(sum1-east)",
            height=25, depth=25, width=2),
    to_Conv("conv5", 3, 256, offset="(0,0,0)",
            to="(relu5-east)", height=35, depth=35, width=4),
    to_BN("bn5", offset="(0,0,0)", to="(conv5-east)",
          height=25, depth=25, width=2, caption="BatchNorm5"),

    to_connection("sum1", "relu5"),

    to_ReLU("relu6", offset="(2,0,0)", to="(bn5-east)",
            height=25, depth=25, width=2),
    to_Conv("conv6", 3, 256, offset="(0,0,0)",
            to="(relu6-east)", height=35, depth=35, width=4),
    to_BN("bn6", offset="(0,0,0)", to="(conv6-east)",
          height=25, depth=25, width=2, caption="BatchNorm6"),

    to_connection("bn5", "relu6"),

    to_Pool("pool2", offset="(1.5,0,0)", to="(bn6-east)", height=40,
            depth=30, width=1, caption="MaxPool2"),  # 2x1
    # to_connection("bn1", "pool1"),

    to_connection("bn6", "pool2"),

    to_Sum("sum2", offset="(1,0,0)", to="(pool2-east)", radius=2, opacity=0.6),

    to_connection("pool2", "sum2"),
    to_skip("relu5", "sum2", pos_of=1.5, pos_to=7),

    # 512 Size Block

    to_ReLU("relu7", offset="(1,0,0)", to="(sum2-east)",
            height=25, depth=25, width=2),
    to_Conv("conv7", 3, 512, offset="(0,0,0)",
            to="(relu7-east)", height=35, depth=35, width=4),
    to_BN("bn7", offset="(0,0,0)", to="(conv7-east)",
          height=25, depth=25, width=2, caption="BatchNorm7"),

    to_connection("sum2", "relu7"),

    to_ReLU("relu8", offset="(2,0,0)", to="(bn7-east)",
            height=25, depth=25, width=2),
    to_Conv("conv8", 3, 512, offset="(0,0,0)",
            to="(relu8-east)", height=35, depth=35, width=4),
    to_BN("bn8", offset="(0,0,0)", to="(conv8-east)",
          height=25, depth=25, width=2, caption="BatchNorm8"),

    to_connection("bn7", "relu8"),

    to_Pool("pool3", offset="(1.5,0,0)", to="(bn8-east)", height=40,
            depth=30, width=1, caption="MaxPool3"),  # 2x1

    to_connection("bn8", "pool3"),

    to_Sum("sum3", offset="(1,0,0)", to="(pool3-east)", radius=2, opacity=0.6),

    to_connection("pool3", "sum3"),
    to_skip("relu7", "sum3", pos_of=1.5, pos_to=7),

    # 728 Size Block

    to_ReLU("relu9", offset="(1,0,0)", to="(sum3-east)",
            height=25, depth=25, width=2),
    to_Conv("conv9", 3, 728, offset="(0,0,0)",
            to="(relu9-east)", height=35, depth=35, width=4),
    to_BN("bn9", offset="(0,0,0)", to="(conv9-east)",
          height=25, depth=25, width=2, caption="BatchNorm9"),

    to_connection("sum3", "relu9"),

    to_ReLU("relu10", offset="(2,0,0)", to="(bn9-east)",
            height=25, depth=25, width=2),
    to_Conv("conv10", 3, 728, offset="(0,0,0)",
            to="(relu10-east)", height=35, depth=35, width=4),
    to_BN("bn10", offset="(0,0,0)", to="(conv10-east)",
          height=25, depth=25, width=2, caption="BatchNorm10"),

    to_connection("bn9", "relu10"),

    to_Pool("pool4", offset="(1.5,0,0)", to="(bn10-east)", height=40,
            depth=30, width=1, caption="MaxPool4"),  # 2x1

    to_connection("bn10", "pool4"),

    to_Sum("sum4", offset="(2,0,0)", to="(pool4-east)", radius=2, opacity=0.6),

    to_connection("pool4", "sum4"),
    to_skip("relu9", "sum4", pos_of=1.5, pos_to=7),

    # 1024 seperable
    to_Conv("conv11", 3, 1024, offset="(2,0,0)",
            to="(sum4-east)", height=30, depth=30, width=4.5),
    to_BN("bn11", offset="(0,0,0)", to="(conv11-east)",
          height=25, depth=25, width=2, caption="BatchNorm11"),
    to_ReLU("relu11", offset="(0,0,0)", to="(bn11-east)",
            height=25, depth=25, width=2),

    to_connection("sum4", "conv11"),

    to_Pool("gap1", offset="(1.5,0,0)", to="(relu11-east)", height=30,
            depth=30, width=1, caption="GlobalAveragePooling2D"),

    to_SoftMax("sigmoid", 2, offset="(1.5,0,0)",
               to="(gap1-east)", caption="Dense (Sigmoid)"),

    to_connection("gap1", "sigmoid"),

    to_end()
]


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')


if __name__ == '__main__':
    main()
