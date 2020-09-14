
# %matplotlib inline
import matplotlib.pyplot as plt
import random
import os, sys
import torch


def getmisclassifiedImage(model, device, aTestLoader):
  misclassified = []
  misclassified_pred = []
  misclassified_target = []
  misclassfiled_list = []
  model.eval()

  with torch.no_grad():
    for data, target in aTestLoader:
      data, target = data.to(device), target.to(device)
     
      output = model(data)
      pred = output.argmax(dim =1, keepdim =True)
      
      list_misclassified = (pred.eq(target.view_as(pred)) == False)
      batch_misclassified = data[list_misclassified]
      batch_mis_pred = pred[list_misclassified]
      batch_mis_target = target.view_as(pred)[list_misclassified]

      misclassified.append(batch_misclassified)
      misclassified_pred.append(batch_mis_pred)
      misclassified_target.append(batch_mis_target)
                                  
  # group all the batched together
  
  misclassified = torch.cat(misclassified)
  misclassified_pred = torch.cat(misclassified_pred)
  misclassified_target = torch.cat(misclassified_target)  
                                
 
  misclassfiled_list.append(misclassified)
  misclassfiled_list.append(misclassified_pred)
  misclassfiled_list.append(misclassified_pred)

  return list(map(lambda x, y, z: (x, y, z), misclassified, misclassified_pred, misclassified_target))

def Plot_misclassifed(model, device, aTestLoader):
    plt.style.use("dark_background")
    misclassified = getmisclassifiedImage(model, device, aTestLoader)
    num_images = 25
    fig = plt.figure(figsize=(12, 12))
    for idx, (image, pred, target) in enumerate(random.choices(misclassified, k=num_images)):
        image, pred, target = image.cpu().numpy(), pred.cpu(), target.cpu()
        ax = fig.add_subplot(5, 5, idx+1)
        ax.axis('off')
        ax.set_title('target {}\npred {}'.format(target.item(), pred.item()), fontsize=12)
        ax.imshow(image.squeeze())

    filepath = None
    if (sys.path[0] != ''):
      filepath = os.path.join(sys.path[0], 'graphs/evas6_misclassified.png')
    else:
      filepath = os.path.join(os.getcwd(), 'graphs/evas6_misclassified.png')

    if os.path.exists(filepath):
        os.remove(filepath) #this deletes the file
    fig.savefig(filepath,dpi=150)
    plt.show()


def PlotValidationGraph(anApproachDicts):
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(11, 9))

    for label, approach_dict in anApproachDicts.items():
        print ("PlotValidationGraph: test_acc: ", approach_dict['test_acc'])
        plt.plot(approach_dict['test_acc'],label = label)

    plt.title('Validation Accuracy vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()


    filepath = None
    if (sys.path[0] != ''):
      filepath = os.path.join(sys.path[0], 'graphs/evas6-acc.png')
    else:
      filepath = os.path.join(os.getcwd(), 'graphs/evas6-acc.png')

    if os.path.exists(filepath):
        os.remove(filepath) #this deletes the file
    fig.savefig(filepath,dpi=150)
    plt.show()

    # from google.colab import files
    # files.download(os.path.join(sys.path[0],'/graphs/evas6-acc.png'))


def PlotLossGraph(anApproachDicts):
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(11, 9))
    # from google.colab import files

    for label, approach_dict in anApproachDicts.items():
        plt.plot(approach_dict['test_losses'],label = label)
        # files.download(os.path.join(sys.path[0],'models/EVAS6-' + label + '.pkl'))


    plt.title('Validation Loss vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    filepath = None
    if (sys.path[0] != ''):
      filepath = os.path.join(sys.path[0], 'graphs/evas6-loss.png')
    else:
      filepath = os.path.join(os.getcwd(), 'graphs/evas6-loss.png')

    if os.path.exists(filepath):
        os.remove(filepath) #this deletes the file

    fig.savefig(filepath,dpi=150)
    # files.download(os.path.join(sys.path[0],'graphs/evas6-loss.png'))

