import os
import time
import xml.dom.minidom


BATCH_SIZE = 500
CUR_DIR = os.getcwd()
DAC_CONTEST = os.path.join(CUR_DIR, '../')
IMG_DIR = os.path.join(CUR_DIR, './images')
OVERLAY_DIR = os.path.join(DAC_CONTEST, './overlay')
RESULT = os.path.join(CUR_DIR, './result')
TIME_DIR = os.path.join(RESULT, './time')
COORD_DIR = os.path.join(RESULT, './coordinate')
XML_PATH = os.path.join(RESULT, './xml')

    
# Get image name list
def get_image_names():
    names_temp = [f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')]
    names_temp.sort(key= lambda x:int(x[:-4]))
    return names_temp


# Process the images in batches, may help when write to XML
def get_image_batch():
    image_list = get_image_names()
    batches = list()
    for i in range(0, len(image_list), BATCH_SIZE):
        batches.append(image_list[i:i+BATCH_SIZE])
    return batches


# Get image paths in batches
def get_image_path(image_name):
    return os.path.join(IMG_DIR, image_name)


# Return a batch of image dir  when `send` is called
class Agent:
    def __init__(self, teamname):
        self.batch_count = 0
        self.dac_contest = DAC_CONTEST
        self.img_dir = IMG_DIR
        self.overlay_dir = OVERLAY_DIR
        self.overlay_dir_team = OVERLAY_DIR + '/' + teamname
        self.result = RESULT
        self.time_dir = TIME_DIR
        self.coord_dir = COORD_DIR
        self.xml_path = XML_PATH
        self.coord_team = COORD_DIR + '/' + teamname
        self.xml_team = XML_PATH + '/' + teamname
        self.contestant = DAC_CONTEST + '/' + teamname
        folder_list = [self.dac_contest, self.img_dir, self.overlay_dir,
                       self.overlay_dir_team,
                       self.result,
                       self.time_dir, self.coord_dir, self.xml_path,
                       self.coord_team, self.xml_team, self.contestant]
        for folder in folder_list:
            if not os.path.isdir(folder):
                os.mkdir(folder)
        self.img_list = get_image_names()
        self.img_batch = get_image_batch()

    def send(self, interval_time, batches):
        time.sleep(interval_time)
        tmp = batches[self.batch_count]
        self.batch_count += 1
        return tmp

    def reset_batch_count(self):
        self.batch_count = 0

    def write(self, t_batch, total_img, teamname):
        fps = total_img / t_batch
        with open(self.time_dir + '/' + teamname + '.txt', 'a+') as f:
            f.write("\n" + teamname + " Frames per second: " +
                    str(fps) + '\n')

    def save_results_xml(self, result_rectangle):
        if len(result_rectangle) != len(self.img_list):
            raise ValueError("Result length not equal to number of images.")
        for i in range(len(self.img_list)):
            doc = xml.dom.minidom.Document()
            root = doc.createElement('annotation')

            doc.appendChild(root)
            name_e = doc.createElement('filename')
            name_t = doc.createTextNode(self.img_list[i])
            name_e.appendChild(name_t)
            root.appendChild(name_e)

            size_e = doc.createElement('size')
            node_width = doc.createElement('width')
            node_width.appendChild(doc.createTextNode("640"))
            node_length = doc.createElement('length')
            node_length.appendChild(doc.createTextNode("360"))
            size_e.appendChild(node_width)
            size_e.appendChild(node_length)
            root.appendChild(size_e)

            object_node = doc.createElement('object')
            node_name = doc.createElement('name')
            node_name.appendChild(doc.createTextNode("NotCare"))
            node_bnd_box = doc.createElement('bndbox')
            node_bnd_box_xmin = doc.createElement('xmin')
            node_bnd_box_xmin.appendChild(
                doc.createTextNode(str(result_rectangle[i][0])))
            node_bnd_box_xmax = doc.createElement('xmax')
            node_bnd_box_xmax.appendChild(
                doc.createTextNode(str(result_rectangle[i][1])))
            node_bnd_box_ymin = doc.createElement('ymin')
            node_bnd_box_ymin.appendChild(
                doc.createTextNode(str(result_rectangle[i][2])))
            node_bnd_box_ymax = doc.createElement('ymax')
            node_bnd_box_ymax.appendChild(
                doc.createTextNode(str(result_rectangle[i][3])))
            node_bnd_box.appendChild(node_bnd_box_xmin)
            node_bnd_box.appendChild(node_bnd_box_xmax)
            node_bnd_box.appendChild(node_bnd_box_ymin)
            node_bnd_box.appendChild(node_bnd_box_ymax)

            object_node.appendChild(node_name)
            object_node.appendChild(node_bnd_box)
            root.appendChild(object_node)

            file_name = self.img_list[i].replace('jpg', 'xml')
            with open(self.xml_team + "/" + file_name, 'w') as fp:
                doc.writexml(fp, indent='\t', addindent='\t',
                             newl='\n', encoding="utf-8")
