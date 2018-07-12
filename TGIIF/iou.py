import os
import xml.dom.minidom as dom

XML_PATH = "/home/xilinx/jupyter_notebooks/dac_2018/xml"

# bbox is in format of [xmin, ymin, xmax, ymax]
def ground_truth(index):
    doc = dom.parse(os.path.join(XML_PATH, "{}.xml".format(index)))
    coor = []
    coor.append(doc.getElementsByTagName("xmin")[0].childNodes[0].data)
    coor.append(doc.getElementsByTagName("xmax")[0].childNodes[0].data)
    coor.append(doc.getElementsByTagName("ymin")[0].childNodes[0].data)
    coor.append(doc.getElementsByTagName("ymax")[0].childNodes[0].data)
    coor = [int(c) for c in coor]
    return coor

def iou(detect, truth):
    area_detect = (detect[1]-detect[0]+1)*(detect[3]-detect[2]+1)
    area_truth = (truth[1]-truth[0]+1)*(truth[3]-truth[2]+1)
    overlap_coor = [max(detect[0], truth[0]), min(detect[1], truth[1]),
                    max(detect[2], truth[2]), min(detect[3], truth[3])]
    if (overlap_coor[0] >= overlap_coor[1] or overlap_coor[2] >= overlap_coor[3]):
        return 0
    else:
        area_overlap = (overlap_coor[1]-overlap_coor[0]+1)*(overlap_coor[3]-overlap_coor[2]+1)
        assert(area_overlap > 0)
        return float(area_overlap) / (area_detect + area_truth - area_overlap) 
