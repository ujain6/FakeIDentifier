import cv2
import operator
import sys
import glob

#jgarcia23


def find_best_match(templates, test_img):
    # empty list of dictionaries to store template images
    template_list_of_dict = []
    # empty dict to store template images
    template_dictionary = {}

    # image to test
    test_image = cv2.imread(test_img)

    # Convert to grayscale
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

    # make a list of all template images from a directory
    template_files = glob.glob(templates)

    for curr_file in template_files:

        image = cv2.imread(curr_file, 0)
        if image is not None:
            template_list_of_dict.append({curr_file: image})
            result = cv2.matchTemplate(test_image, image, cv2.TM_CCOEFF)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            template_dictionary.update({curr_file: max_val})

    threshold = max(template_dictionary.items(), key=operator.itemgetter(1))[0]
    state = threshold.strip('id_templates/''.jpeg''1')

    return state, threshold


def display_result(template_files, threshold,test_image):
    template_files = glob.glob(template_files)

    for i in template_files:
        if threshold == i:

            temp = cv2.imread(i, 0)

            # Find template
            res = cv2.matchTemplate(test_image, temp, cv2.TM_CCOEFF)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            top_left = max_loc
            h, w = temp.shape
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv2.rectangle(test_image, top_left, bottom_right, (0, 0, 255), 4)

            # # Show result
            cv2.imshow("Template", temp)
            cv2.imshow("Result", test_image)
            #
            cv2.moveWindow("Template", 10, 50);
            cv2.moveWindow("Result", 150, 50);
            # #
            cv2.waitKey(5000)
            cv2.destroyAllWindows()


def process_args(args):
    # parse commandline inputs
    # if len(args) < 2:
    #     print("USAGE: python3 state_det.py <path to templates> <path to image>")
    #     return None

    # templates_files = args[1]
    # test_image = args[2]

    templates = 'id_templates/*.jpeg'
    test_img = 'driver3.jpg'

    result, threshold = find_best_match(templates, test_img)
    # print best match
    print('this is a ', result, ' ID')

    img = cv2.imread(test_img, 0)
    display_result(templates, threshold,img)


def main():
    process_args(sys.argv)


if __name__ == '__main__':
    main()
