def forward(self, x):
    conv1_conv = self.conv1.conv(x);  x = None
    conv1_relu = self.conv1.relu(conv1_conv);  conv1_conv = None
    maxpool1 = self.maxpool1(conv1_relu);  conv1_relu = None
    conv2_conv = self.conv2.conv(maxpool1);  maxpool1 = None
    conv2_relu = self.conv2.relu(conv2_conv);  conv2_conv = None
    conv3_conv = self.conv3.conv(conv2_relu);  conv2_relu = None
    conv3_relu = self.conv3.relu(conv3_conv);  conv3_conv = None
    maxpool2 = self.maxpool2(conv3_relu);  conv3_relu = None
    inception3a_branch1_conv = self.inception3a.branch1.conv(maxpool2)
    inception3a_branch1_relu = self.inception3a.branch1.relu(inception3a_branch1_conv);  inception3a_branch1_conv = None
    inception3a_branch2_0_conv = getattr(self.inception3a.branch2, "0").conv(maxpool2)
    inception3a_branch2_0_relu = getattr(self.inception3a.branch2, "0").relu(inception3a_branch2_0_conv);  inception3a_branch2_0_conv = None
    inception3a_branch2_1_conv = getattr(self.inception3a.branch2, "1").conv(inception3a_branch2_0_relu);  inception3a_branch2_0_relu = None
    inception3a_branch2_1_relu = getattr(self.inception3a.branch2, "1").relu(inception3a_branch2_1_conv);  inception3a_branch2_1_conv = None
    inception3a_branch3_0_conv = getattr(self.inception3a.branch3, "0").conv(maxpool2)
    inception3a_branch3_0_relu = getattr(self.inception3a.branch3, "0").relu(inception3a_branch3_0_conv);  inception3a_branch3_0_conv = None
    inception3a_branch3_1_conv = getattr(self.inception3a.branch3, "1").conv(inception3a_branch3_0_relu);  inception3a_branch3_0_relu = None
    inception3a_branch3_1_relu = getattr(self.inception3a.branch3, "1").relu(inception3a_branch3_1_conv);  inception3a_branch3_1_conv = None
    inception3a_branch4_0 = getattr(self.inception3a.branch4, "0")(maxpool2);  maxpool2 = None
    inception3a_branch4_1_conv = getattr(self.inception3a.branch4, "1").conv(inception3a_branch4_0);  inception3a_branch4_0 = None
    inception3a_branch4_1_relu = getattr(self.inception3a.branch4, "1").relu(inception3a_branch4_1_conv);  inception3a_branch4_1_conv = None
    cat = torch.cat([inception3a_branch1_relu, inception3a_branch2_1_relu, inception3a_branch3_1_relu, inception3a_branch4_1_relu], 1);  inception3a_branch1_relu = inception3a_branch2_1_relu = inception3a_branch3_1_relu = inception3a_branch4_1_relu = None
    return cat



def forward(self,x):
    x = self.b1(x)
    b1 = self.b2_1.remote(x)
    b2 = self.b2_2.remote(x)
    b3 = self.b2_3.remote(x)
    b4 = self.b2_4.remote(x)
    x = self.b3(b1, b2, b3, b4)
    return x

def b1(self,x):
    conv1_conv = self.conv1.conv(x);  x = None
    conv1_relu = self.conv1.relu(conv1_conv);  conv1_conv = None
    maxpool1 = self.maxpool1(conv1_relu);  conv1_relu = None
    conv2_conv = self.conv2.conv(maxpool1);  maxpool1 = None
    conv2_relu = self.conv2.relu(conv2_conv);  conv2_conv = None
    conv3_conv = self.conv3.conv(conv2_relu);  conv2_relu = None
    conv3_relu = self.conv3.relu(conv3_conv);  conv3_conv = None
    maxpool2 = self.maxpool2(conv3_relu);  conv3_relu = None
    return maxpool2

@ray.remote
def b2_1(self,maxpool2):
    inception3a_branch1_conv = self.inception3a.branch1.conv(maxpool2)
    inception3a_branch1_relu = self.inception3a.branch1.relu(inception3a_branch1_conv); inception3a_branch1_conv = None
    return inception3a_branch1_relu

@ray.remote
def b2_2(self,maxpool2):
    inception3a_branch2_0_conv = getattr(self.inception3a.branch2, "0").conv(maxpool2)
    inception3a_branch2_0_relu = getattr(self.inception3a.branch2, "0").relu(inception3a_branch2_0_conv);  inception3a_branch2_0_conv = None
    inception3a_branch2_1_conv = getattr(self.inception3a.branch2, "1").conv(inception3a_branch2_0_relu);  inception3a_branch2_0_relu = None
    inception3a_branch2_1_relu = getattr(self.inception3a.branch2, "1").relu(inception3a_branch2_1_conv);  inception3a_branch2_1_conv = None
    return inception3a_branch2_1_relu

@ray.remote
def b2_3(self,maxpool2):
    inception3a_branch3_0_conv = getattr(self.inception3a.branch3, "0").conv(maxpool2)
    inception3a_branch3_0_relu = getattr(self.inception3a.branch3, "0").relu(inception3a_branch3_0_conv);  inception3a_branch3_0_conv = None
    inception3a_branch3_1_conv = getattr(self.inception3a.branch3, "1").conv(inception3a_branch3_0_relu);  inception3a_branch3_0_relu = None
    inception3a_branch3_1_relu = getattr(self.inception3a.branch3, "1").relu(inception3a_branch3_1_conv);  inception3a_branch3_1_conv = None
    return inception3a_branch3_1_relu

@ray.remote
def b2_4(self,maxpool2):
    inception3a_branch4_0 = getattr(self.inception3a.branch4, "0")(maxpool2);  maxpool2 = None
    inception3a_branch4_1_conv = getattr(self.inception3a.branch4, "1").conv(inception3a_branch4_0);  inception3a_branch4_0 = None
    inception3a_branch4_1_relu = getattr(self.inception3a.branch4, "1").relu(inception3a_branch4_1_conv);  inception3a_branch4_1_conv = None
    return inception3a_branch4_1_relu

def b3(self,inception3a_branch1_relu, inception3a_branch2_1_relu , inception3a_branch3_1_relu, inception3a_branch4_1_relu):
    cat = torch.cat([inception3a_branch1_relu, inception3a_branch2_1_relu, inception3a_branch3_1_relu, inception3a_branch4_1_relu], 1);  inception3a_branch1_relu = inception3a_branch2_1_relu = inception3a_branch3_1_relu = inception3a_branch4_1_relu = None
    return cat