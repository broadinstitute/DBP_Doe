# Imports
from fpdf import FPDF, HTMLMixin
from datetime import datetime
import time
from pip._internal.operations.freeze import freeze
import subprocess
import os
from skimage import io
import FromScratch


#Defining variables imported from FromScratch file 
# is this a good way to import the variables?
Network = FromScratch.Network
model_name = FromScratch.model_name
model_path = FromScratch.model_path
training_source = FromScratch.training_source
train_generator = FromScratch.train_generator
number_of_epochs = FromScratch.number_of_epochs
add_gaussian_blur = FromScratch.add_gaussian_blur
add_linear_contrast = FromScratch.add_linear_contrast
add_additive_gaussian_noise = FromScratch.add_additive_gaussian_noise
augmenters = FromScratch.augmenters
add_elastic_deform = FromScratch.add_elastic_deform
use_default_advanced_parameters = FromScratch.use_default_advanced_parameters
batch_size = FromScratch.batch_size
patch_size = FromScratch.patch_size
training_target = FromScratch.training_target





#Time stamp
start = time.time()
dt = time.time() - start
mins, sec = divmod(dt, 60)
hour, mins = divmod(mins, 60)

# A function for the PDF export 
def pdf_export(trained = False, augmentation = False, pretrained_model = False):
  class MyFPDF(FPDF, HTMLMixin):
    pass

  pdf = MyFPDF()
  pdf.add_page()
  pdf.set_right_margin(-1)
  pdf.set_font("Arial", size = 11, style='B')

  day = datetime.now()
  datetime_str = str(day)[0:10]

  Header = 'Training report for '+Network+' model ('+model_name+')\nDate: '+datetime_str
  pdf.multi_cell(180, 5, txt = Header, align = 'L')
  pdf.ln(1)

  # add another cell
  if trained:
    training_time = "Training time: "+str(hour)+ "hour(s) "+str(mins)+"min(s) "+str(round(sec))+"sec(s)"
    pdf.cell(190, 5, txt = training_time, ln = 1, align='L')
  pdf.ln(1)

  Header_2 = 'Information for your materials and methods:'
  pdf.cell(190, 5, txt=Header_2, ln=1, align='L')
  pdf.ln(1)

  all_packages = ''
  for requirement in freeze(local_only=True):
    all_packages = all_packages+requirement+', '
  

  #Main Packages
  main_packages = ''
  version_numbers = []
  for name in ['tensorflow','numpy','keras']:
    find_name=all_packages.find(name)
    main_packages = main_packages+all_packages[find_name:all_packages.find(',',find_name)]+', '
    #Version numbers only here:
    version_numbers.append(all_packages[find_name+len(name)+2:all_packages.find(',',find_name)])

  try:
    cuda_version = subprocess.run(["nvcc","--version"],stdout=subprocess.PIPE)
    cuda_version = cuda_version.stdout.decode('utf-8')
    cuda_version = cuda_version[cuda_version.find(', V')+3:-1]
  except:
    cuda_version = ' - No cuda found - '
  try:
    gpu_name = subprocess.run(["nvidia-smi"],stdout=subprocess.PIPE)
    gpu_name = gpu_name.stdout.decode('utf-8')
    gpu_name = gpu_name[gpu_name.find('Tesla'):gpu_name.find('Tesla')+10]
  except:
    gpu_name = ' - No GPU found - '


  if os.path.isdir(training_source):
    shape = io.imread(training_source+'/'+os.listdir(training_source)[0]).shape
  elif os.path.isfile(training_source):
    shape = io.imread(training_source).shape
  else:
    print('Cannot read training data.')

  dataset_size = len(train_generator)

  text = 'The '+Network+' model was trained from scratch for '+str(number_of_epochs)+' epochs on '+str(dataset_size)+' paired image patches (image dimensions: '+str(shape)+', patch size: ('+str(patch_size)+') with a batch size of '+str(batch_size)+' and a '+loss_function+' loss function, using the '+Network+' ZeroCostDL4Mic notebook (v '+Notebook_version[0]+') (von Chamier & Laine et al., 2020). Key python packages used include tensorflow (v '+version_numbers[0]+'), keras (v '+version_numbers[2]+'), numpy (v '+version_numbers[1]+'), cuda (v '+cuda_version+'). The training was accelerated using a '+gpu_name+'GPU.'

  if pretrained_model:
    text = 'The '+Network+' model was trained for '+str(number_of_epochs)+' epochs on '+str(dataset_size)+' paired image patches (image dimensions: '+str(shape)+', patch_size: '+str(patch_size)+') with a batch size of '+str(batch_size)+' and a '+loss_function+' loss function, using the '+Network+' ZeroCostDL4Mic notebook (v '+Notebook_version[0]+') (von Chamier & Laine et al., 2020). The model was retrained from a pretrained model. Key python packages used include tensorflow (v '+version_numbers[0]+'), keras (v '+version_numbers[2]+'), numpy (v '+version_numbers[1]+'), cuda (v '+cuda_version+'). The training was accelerated using a '+gpu_name+'GPU.'

  pdf.set_font('')
  pdf.set_font_size(10.)
  pdf.multi_cell(190, 5, txt = text, align='L')
  pdf.ln(1)
  pdf.set_font('')
  pdf.set_font('Arial', size = 10, style = 'B')
  pdf.cell(28, 5, txt='Augmentation: ', ln=0)
  pdf.set_font('')

  if augmentation:
    aug_text = 'The dataset was augmented by'
    if add_gaussian_blur == True:
      aug_text = aug_text+'\n- gaussian blur'
    if add_linear_contrast == True:
      aug_text = aug_text+'\n- linear contrast'
    if add_additive_gaussian_noise == True:
      aug_text = aug_text+'\n- additive gaussian noise'
    if augmenters != '':
      aug_text = aug_text+'\n- imgaug augmentations: '+augmenters
    if add_elastic_deform == True:
      aug_text = aug_text+'\n- elastic deformation'
  else:
    aug_text = 'No augmentation was used for training.'
  pdf.multi_cell(190, 5, txt=aug_text, align='L')
  pdf.ln(1)
  pdf.set_font('Arial', size = 11, style = 'B')
  pdf.cell(180, 5, txt = 'Parameters', align='L', ln=1)
  pdf.set_font('')
  pdf.set_font_size(10.)
  if use_default_advanced_parameters:
    pdf.cell(200, 5, txt='Default Advanced Parameters were enabled')
  pdf.cell(200, 5, txt='The following parameters were used for training:')
  pdf.ln(1)
  html = """
  <table width=60% style="margin-left:0px;">
    <tr>
      <th width = 50% align="left">Parameter</th>
      <th width = 50% align="left">Value</th>
    </tr>
    <tr>
      <td width = 50%>number_of_epochs</td>
      <td width = 50%>{0}</td>
    </tr>
    <tr>
      <td width = 50%>batch_size</td>
      <td width = 50%>{1}</td>
    </tr>
    <tr>
      <td width = 50%>patch_size</td>
      <td width = 50%>{2}</td>
    </tr>
    <tr>
      <td width = 50%>image_pre_processing</td>
      <td width = 50%>{3}</td>
    </tr>
    <tr>
      <td width = 50%>validation_split_in_percent</td>
      <td width = 50%>{4}</td>
    </tr>
      <tr>
      <td width = 50%>downscaling_in_xy</td>
      <td width = 50%>{5}</td>
    </tr>
      <tr>
      <td width = 50%>binary_target</td>
      <td width = 50%>{6}</td>
    </tr>
    <tr>
      <td width = 50%>loss_function</td>
      <td width = 50%>{7}</td>
    </tr>
    <tr>
      <td width = 50%>metrics</td>
      <td width = 50%>{8}</td>
    </tr>
    <tr>
      <td width = 50%>optimizer</td>
      <td width = 50%>{9}</td>
    </tr>
    <tr>
      <td width = 50%>checkpointing_period</td>
      <td width = 50%>{10}</td>
    </tr>
    <tr>
      <td width = 50%>save_best_only</td>
      <td width = 50%>{11}</td>
    </tr>
    <tr>
      <td width = 50%>resume_training</td>
      <td width = 50%>{12}</td>
    </tr>
  </table>
  """.format(number_of_epochs,batch_size,str(patch_size[0])+'x'+str(patch_size[1])+'x'+str(patch_size[2]),image_pre_processing, validation_split_in_percent, downscaling_in_xy, str(binary_target), loss_function, metrics, optimizer, checkpointing_period, str(save_best_only), str(resume_training))
  pdf.write_html(html)

  #pdf.multi_cell(190, 5, txt = text_2, align='L')
  pdf.set_font("Arial", size = 11, style='B')
  pdf.ln(1)
  pdf.cell(190, 5, txt = 'Training Dataset', align='L', ln=1)
  pdf.set_font('')
  pdf.set_font('Arial', size = 10, style = 'B')
  pdf.cell(30, 5, txt= 'Training_source:', align = 'L', ln=0)
  pdf.set_font('')
  pdf.multi_cell(170, 5, txt = training_source, align = 'L')
  pdf.ln(1)
  pdf.set_font('')
  pdf.set_font('Arial', size = 10, style = 'B')
  pdf.cell(28, 5, txt= 'Training_target:', align = 'L', ln=0)
  pdf.set_font('')
  pdf.multi_cell(170, 5, txt = training_target, align = 'L')
  pdf.ln(1)
  pdf.set_font('')
  pdf.set_font('Arial', size = 10, style = 'B')
  pdf.cell(21, 5, txt= 'Model Path:', align = 'L', ln=0)
  pdf.set_font('')
  pdf.multi_cell(170, 5, txt = model_path+'/'+model_name, align = 'L')
  pdf.ln(1)
  

  pdf.output(model_path+'/'+model_name+'/'+model_name+'_training_report.pdf')

  print('------------------------------')
  print('PDF report exported in '+model_path+'/'+model_name+'/')


# check whether you need the QC PDF export and retain this function based on that 
def qc_pdf_export():
  class MyFPDF(FPDF, HTMLMixin):
    pass

  pdf = MyFPDF()
  pdf.add_page()
  pdf.set_right_margin(-1)
  pdf.set_font("Arial", size = 11, style='B')

  Network = 'U-Net 3D'

  day = datetime.now()
  datetime_str = str(day)[0:10]

  Header = 'Quality Control report for '+Network+' model ('+qc_model_name+')\nDate: '+datetime_str
  pdf.multi_cell(180, 5, txt = Header, align = 'L')
  pdf.ln(1)

  all_packages = ''
  for requirement in freeze(local_only=True):
    all_packages = all_packages+requirement+', '

  pdf.set_font('')
  pdf.set_font('Arial', size = 11, style = 'B')
  pdf.ln(2)
  pdf.cell(190, 5, txt = 'Loss curves', ln=1, align='L')
  pdf.ln(1)
  if os.path.exists(os.path.join(qc_model_path,qc_model_name,'Quality Control')+'/lossCurvePlots.png'):
    exp_size = io.imread(os.path.join(qc_model_path,qc_model_name,'Quality Control')+'/lossCurvePlots.png').shape
    pdf.image(os.path.join(qc_model_path,qc_model_name,'Quality Control')+'/lossCurvePlots.png', x = 11, y = None, w = round(exp_size[1]/8), h = round(exp_size[0]/8))
  else:
    pdf.set_font('')
    pdf.set_font('Arial', size=10)
    pdf.multi_cell(190, 5, txt='If you would like to see the evolution of the loss function during training please play the first cell of the QC section in the notebook.')
  pdf.ln(2)
  pdf.set_font('')
  pdf.set_font('Arial', size = 10, style = 'B')
  pdf.ln(3)
  pdf.cell(80, 5, txt = 'Example Quality Control Visualisation', ln=1)
  pdf.ln(1)
  exp_size = io.imread(os.path.join(qc_model_path,qc_model_name,'Quality Control')+'/QC_example_data.png').shape
  pdf.image(os.path.join(qc_model_path,qc_model_name,'Quality Control')+'/QC_example_data.png', x = 5, y = None, w = round(exp_size[1]/12), h = round(exp_size[0]/8))
  pdf.ln(1)
  pdf.set_font('')
  pdf.set_font('Arial', size = 11, style = 'B')
  pdf.ln(1)
  pdf.cell(180, 5, txt = 'IoU threshold optimisation', align='L', ln=1)
  pdf.set_font('')
  pdf.set_font_size(10.)
  pdf.ln(1)
  pdf.cell(120, 5, txt='Highest IoU is {:.4f} with a threshold of {}'.format(best_iou, best_thresh), align='L', ln=1)
  pdf.ln(2)
  exp_size = io.imread(os.path.join(qc_model_path,qc_model_name,'Quality Control')+'/QC_IoU_analysis.png').shape
  pdf.image(os.path.join(qc_model_path,qc_model_name,'Quality Control')+'/QC_IoU_analysis.png', x=16, y=None, w = round(exp_size[1]/6), h = round(exp_size[0]/6))
  pdf.ln(1)
  pdf.set_font('')
  pdf.set_font_size(10.)
  ref_1 = 'References:\n - ZeroCostDL4Mic: von Chamier, Lucas & Laine, Romain, et al. "Democratising deep learning for microscopy with ZeroCostDL4Mic." Nature Communications (2021).'
  pdf.multi_cell(190, 5, txt = ref_1, align='L')
  pdf.ln(1)
  ref_2 = '- Unet 3D: Çiçek, Özgün, et al. "3D U-Net: learning dense volumetric segmentation from sparse annotation." International conference on medical image computing and computer-assisted intervention. Springer, Cham, 2016.'
  pdf.multi_cell(190, 5, txt = ref_2, align='L')
  pdf.ln(1)

  pdf.ln(3)
  reminder = 'To find the parameters and other information about how this model was trained, go to the training_report.pdf of this model which should be in the folder of the same name.'

  pdf.set_font('Arial', size = 11, style='B')
  pdf.multi_cell(190, 5, txt=reminder, align='C')
  pdf.ln(1)

  pdf.output(os.path.join(qc_model_path,qc_model_name,'Quality Control')+'/'+qc_model_name+'_QC_report.pdf')

  print('------------------------------')
  print('QC PDF report exported in '+os.path.join(qc_model_path,qc_model_name,'Quality Control')+'/')