import subprocess
#used to call C++, arguments can be changed here
def generate(ycb_dir,file_name, IND, table_to_cam,search_object):
    directory = ycb_dir +'data/'+file_name+'/'+"{:06d}".format(IND)+"-color.png"
    #label is used to do background subtraction
    label = ycb_dir+ 'data/'+file_name+'/'+"{:06d}".format(IND)+"-label.png"
    command = ["rosrun", "color_only", "color_render", directory, str(IND), \
        str(table_to_cam[0,0]),str(table_to_cam[0,1]),str(table_to_cam[0,2]),str(table_to_cam[0,3]),\
        str(table_to_cam[1,0]),str(table_to_cam[1,1]),str(table_to_cam[1,2]),str(table_to_cam[1,3]),\
        str(table_to_cam[2,0]),str(table_to_cam[2,1]),str(table_to_cam[2,2]),str(table_to_cam[2,3]), label,search_object]
    print(' '.join(command))
    test = subprocess.Popen(command, stdout=subprocess.PIPE)
    output = test.communicate()[0]
