# -*- coding = utf-8 =_=
__author__ = '15624959453@163.com'

import os
import sys
import re
from glob import glob
from collections import defaultdict
from pprint import pprint

# DataPreprocess.PrioriExtract
# Parameters:
#     sample_content: 2d array or 2d matrix analogously,
#         store the content of sample info table, which include patientID, sampleID, tumor type and so on
#     pathology_content: 2d array or 2d matrix like, which include patientID, pathology type of WHO and lauren, differentiation type
# Methods:
#     originOfSample: Return map table of tissue type with sampleID, to check which tissue the sample are from
#     patient_with_sample: Return map table of sample with patientID, for the handling to different samples of one patient
#     patient_id_capture: store all patient id in array, which is the main index for connecting the pathology type with mutation information
#     patient_pathologytype: transfer the real pathology type signs to standard label
class PrioriExtract(object):
    def __init__(self, sample_info_filePath, pathology_type_filePath):
        self.sample_content = []
        with open(sample_info_filePath, "r") as f1:
            for line in f1:
                self.sample_content.append(line.strip("\r\n").split("\t"))
        self.pathology_content = []
        with open(pathology_type_filePath, "r") as f2:
            for lineno, line in enumerate(f2):
                if lineno >= 2:
                    data_array = line.strip("\n").split("\t")
                    self.pathology_content.append(data_array)

    def originOfSample(self):
        '''
        :return: dict, with tissue type as keys and sampleID as values
        '''
        res = {'fresh tissue':set(), 'blood':set()}
        for info in self.sample_content:
            res[info[8]].add(info[5])
        return res

    def patient_with_sample(self):
        '''
        :return: dict, with sampleID as keys and map table of tissue type and sampleID as values
        '''
        res, origin_info = defaultdict(dict), self.originOfSample()
        for info in self.sample_content:
            if info[5] in origin_info["fresh tissue"]:
                if "fresh tissue" in res[info[1]]:
                    res[info[1]]["fresh tissue"].append(info[5])
                else:
                    res[info[1]]["fresh tissue"] = [info[5]]
            else:
                if "blood" in res[info[1]]:
                    res[info[1]]["blood"].append(info[5])
                else:
                    res[info[1]]["blood"] = [info[5]]
        return dict(res)

    def patient_id_capture(self):
        '''
        :return: 1d array, all the sample ID included.
        '''
        res = []
        for line in self.sample_content:
            if not res or line[1] != res[-1]:
                res.append(line[1])
        return res

    def patient_pathologytype(self):
        '''
        :return: dict, with patientID as keys and pathology types as values
        '''
        res = defaultdict(list)
        for patient_info in self.pathology_content:
            patient_id = patient_info[2]
            who_level = 0
            for i in range(4):
                if patient_info[3+i]:
                    if patient_info[3+i].find("%") != -1:
                        potential_num = patient_info[3+i].split("%")[0]
                        if potential_num.isdigit():
                            who_level += int(potential_num) / 100.0 * i
                    else:
                        who_level += int(patient_info[3+i]) * i
            diff_level = sum(
                [int(patient_info[8+j].split("%")[0]) / 100.0 * j \
                    if patient_info[8+j].find("%") != -1 else int(patient_info[8+j]) * j for j in range(3) \
                    if patient_info[8+j]]
            )
            lauren_level = sum([int(patient_info[11+p]) * p for p in range(2) if patient_info[11+p]])
            res[patient_id] = [who_level, diff_level, lauren_level]
        return res


# DataPreprocess.Preprocess
# Parameters:
#     tissue_info: dict, with tissue type as keys and sampleID as values
#     patient_info: dict, with sampleID as keys and map table of tissue type and sampleID as values
#     patient_ids: 1d array, all the sample ID included, sort in accordance with ascending index
#     patient_pathInfo: dict, with patientID as keys and pathology types as values
#     sample_with_path: dict, with sampleID as keys and sample fastq file path as values, for detecting the mutation analysis result.
# Methods:
#     filePath_obtain: search the result directory to seek the unique mutation analysis result, to make map table of sampleID with mutation result path.
#     path_from_patient: return the mutation analysis result path by patientID
#     filter_demo: demo routine for mutation filter, seek in the result directory and construct the connection of patientID and mutation file, then screen the gene mutation for each patient in accordance with filter standard.
#     filter_mutation: formal routine for mutation filter, then merge the mutation information with labels, seek in the result directory and construct the connection of patientID and mutation file, then screen the gene mutation for each patient in accordance with filter standard.
#     snvIndel_mut_filter: give a mutation analysis result, decide whether it belongs to blood sample of tissue sample, then filter the mutation gene with specific rules.
class Preprocess(PrioriExtract):
    def __init__(self, full_info, pathology_info):
        PrioriExtract.__init__(self, full_info, pathology_info)
        self.tissue_info = self.originOfSample()
        self.patient_info = self.patient_with_sample()
        self.patient_ids = self.patient_id_capture()
        self.patient_pathInfo = self.patient_pathologytype()
        self.sample_with_path = {}

    def filePath_obtain(self, directory):
        '''
        :param directory: the root of all result
        :return: dict, with sampleID as keys and filepath as values
        '''
        fileList = glob(os.path.join(directory, "WETask*/SNVAndIndel/*/*.dedup.txt"))
        for filePath in fileList:
            sampleName = ".".join(os.path.basename(filePath).split(".")[:-2])
            self.sample_with_path[sampleName] = filePath
        return self.sample_with_path

    def path_from_patient(self, patientID):
        '''
        :param patientID: patient ID, to identify the specific patient.
        :return: dict, with tissue type as keys and sample mutation analysis file as values
        '''
        res = {"blood":[], "fresh tissue":[]}
        if patientID in self.patient_info:
            for tissue_type, samples in self.patient_info[patientID].items():
                for sample in samples:
                    if sample in self.sample_with_path:
                        res[tissue_type].append(self.sample_with_path[sample])
                    else:
                        print("Warning, file path load have not been accomplished.")
        else:
            print("Warning, no memory for patientID {} in patient_infos.".format(patientID))
        return res

    # def filter_demo(self, patient_nums, directory, output_dir):
    #     self.filePath_obtain(directory)
    #     for id in self.patient_ids[:patient_nums]:
    #         samplePaths = self.path_from_patient(id)
    #         control_total_data = []
    #         exper_total_data, exper_total_file = [], os.path.join(output_dir, "patient{}_mutation.table".format(id))

    #         for control_path in samplePaths["blood"]:
    #             filter_data = self.snvIndel_mut_filter(control_path)
    #             control_total_data += filter_data
    #         for exper_path in samplePaths["fresh tissue"]:
    #             filter_data = self.snvIndel_mut_filter(exper_path)
    #             exper_total_data += filter_data

    #         control_total_data = sorted(
    #             control_total_data, key = lambda x: (int(x[0][3:]) if x[0][3:].isdigit() else x[0][3:], int(x[1]))
    #         )
    #         exper_total_data = sorted(
    #             exper_total_data, key = lambda x: (int(x[0][3:]) if x[0][3:].isdigit() else x[0][3:], int(x[1]))
    #         )
    #         control_mut_sites = set(["_".join(line[:3]) for line in control_total_data])
    #         with open(exper_total_file, "w") as f1:
    #             for exper_data in exper_total_data:
    #                 exper_mut_site = "_".join(exper_data[:3])
    #                 if exper_mut_site not in control_mut_sites:
    #                     f1.write("\t".join(exper_data) + "\n")
    #                 # else:
    #                 #     print(exper_mut_sites)
    #     return True

    def filter_mutation(self, patient_nums, directory, output_dir):
        '''
        :param patient_nums: integer, how many patient to filter mutation.
        :param directory: root of all result
        :param output_dir: directory for filter file.
        :return: Boolean
        '''
        self.filePath_obtain(directory)
        mutation_file = os.path.join(output_dir, "mutation_for_1-52.table")
        single_mutation_file = os.path.join(output_dir, "mutation_single_display_for_1-52.table")
        total_mutaion_set = set()
        mutation_matrix = []

        for id in self.patient_ids[:patient_nums]:
            samplePaths = self.path_from_patient(id)
            control_total_data = []
            exper_total_data = []

            # 对于一个病人，将每个样本按照组织类型进行过滤，调用了 snvIndel_mut_filter 方法
            for control_path in samplePaths["blood"]:
                filter_data = self.snvIndel_mut_filter(control_path)
                control_total_data += filter_data
            for exper_path in samplePaths["fresh tissue"]:
                filter_data = self.snvIndel_mut_filter(exper_path)
                exper_total_data += filter_data

            control_total_data = sorted(
                control_total_data, key = lambda x: (int(x[0][3:]) if x[0][3:].isdigit() else x[0][3:], int(x[1]))
            )
            exper_total_data = sorted(
                exper_total_data, key = lambda x: (int(x[0][3:]) if x[0][3:].isdigit() else x[0][3:], int(x[1]))
            )
            control_mut_sites = set(["_".join(line[:3]) for line in control_total_data])
            mutation_matrix.append([id])
            # filter the gene mutation by control background
            for exper_data in exper_total_data:
                exper_mut_site = "_".join(exper_data[:3])
                if exper_mut_site not in control_mut_sites:
                    for gene in exper_data[11].split(","):
                        if gene:
                            mutation_matrix[-1].append(gene)
                            total_mutaion_set.add(gene)

        # store the filter information in hard disk
        total_mutaion_list = list(total_mutaion_set)
        mutation_signs = []
        for lineno, patient in enumerate(mutation_matrix):
            real_mutation, single_mutation = [], set(patient[1:])
            for mutation in total_mutaion_list:
                real_mutation.append("1" if mutation in single_mutation else "0")
            labels = ["", "", ""] if patient[0] not in self.patient_pathInfo else [str(x) for x in self.patient_pathInfo[patient[0]]]
            mutation_signs.append(patient[:1] + real_mutation + labels)
        mutation_signs = [["patient id"] + total_mutaion_list + ["who_level", "diff_level", "lauren_level"]] + mutation_signs
        with open(mutation_file, "w") as f1:
            for line in mutation_signs:
                f1.write("\t".join(line) + "\n")
        with open(single_mutation_file, "w") as f2:
            for lineno, line in enumerate(mutation_matrix):
                line = line[:1] + list(set(line[1:]))
                f2.write("\t".join(line) + "\n")

        return True

    def snvIndel_mut_filter(self, fileName):
        '''
        :param fileName: mutation analysis result file path
        :return: gene table including locus, gene, and some other information
        '''
        sampleName, sample_tissue = ".".join(os.path.basename(fileName).split(".")[:-2]), ""
        if sampleName in self.tissue_info["blood"]:
            sample_tissue = "blood"
        else:
            sample_tissue = "fresh tissue"

        filter_data = []
        # execute the filter standard.
        with open(fileName, "r") as f1:
            for lineno, data in enumerate(f1):
                if lineno >= 1:
                    flag = False
                    line = data.strip().split("\t")
                    line = line + ["" for _ in range(70 - len(line))]

                    if (10 < int(line[5]) + int(line[6]) and 3 <= int(line[6]) and sample_tissue == "fresh tissue") or \
                            (10 < int(line[5]) + int(line[6]) and 1 < int(line[6]) and sample_tissue == "blood"):
                        flag = True
                    if sample_tissue == "fresh tissue" and \
                            10 < int(line[5]) + int(line[6]) and 3 <= int(line[6]) and \
                            float(line[12]) <= 1 and \
                            line[23] not in ["UNKNOWN", ""] and \
                            line[24] not in ["UNKNOWN", ""] and \
                            line[25] not in ["UNKNOWN", ""] and \
                            line[26] not in ["UNKNOWN", ""] and \
                            line[29] not in ["unknown", ""] and \
                            line[32] == "" and line[35] == "" and \
                            (line[37] == "" or (line[37] != "" and line[64] != "")):
                        filter_data.append(line[:5] + [line[21]] + line[23:30])
                    if sample_tissue == "blood" and \
                             10 < int(line[5]) + int(line[6]) and 1 < int(line[6]) and \
                             float(line[12]) <= 1:
                        filter_data.append(line[:5] + [line[21]] + line[23:30])
                            # for gene in line[28].split(","):
                        #     if gene:
                        #         geneSet.add(gene)
        # with open(filterFile, "w") as f2:
        #     for line in filter_data:
        #         f2.write("\t".join(line) + "\n")
        return filter_data


if __name__ == "__main__":
    # geneSet_unique, geneSet_control, geneSet_tumor = set(), geneSet_L3 | geneSet_L4 | geneSet_L5, geneSet_L8 | geneSet_L6 | geneSet_L7
    # geneSet_cross = geneSet_tumor & geneSet_control
    # for gene in geneSet_tumor:
    #     if gene in geneSet_control:
    #         geneSet_unique.add(gene)
    #         if gene.find("TP53") != -1:
    #             print(True)
    #
    # from pprint import pprint
    # pprint(geneSet_unique)
    # print(len(geneSet_unique))

    # sampleInfo_path = "/lustre/users/fangwenzheng/gastricCancer/Reference/full_ver_sample_info.bed"
    sampleInfo_path = "D:\\公司项目_方文征\\胃癌检测项目\\Data\\full_ver_sample_info.bed"
    pathology_info_path = "D:\\公司项目_方文征\\胃癌检测项目\\Data\\pathology_type.txt"
    result_path = "/lustre/common/WebServiceResult/Pro000068"
    outputDir = "/lustre/users/fangwenzheng/gastricCancer/result"

    quantifyFile = "/lustre/common/WebServiceResult/Pro000068/WETask17/SNVAndIndel/EGAR00001274628_FCC2C2NACXX_L3_HUMdjpXAAABAAA-91/EGAR00001274628_FCC2C2NACXX_L3_HUMdjpXAAABAAA-91.dedup.txt"
    filterFile = "/lustre/users/fangwenzheng/gastricCancer/result/filter_for_EGAR00001274628_FCC2C2NACXX_L3_HUMdjpXAAABAAA-91.table"

    pe = PrioriExtract(sampleInfo_path, pathology_info_path)
    sampleProcess = Preprocess(sampleInfo_path, pathology_info_path)
    # sampleProcess.filePath_obtain(result_path)
    # print(sampleProcess.snvIndel_mut_filter(quantifyFile))
    # print(sampleProcess.path_from_patient("211421"))
    sampleProcess.filter_mutation(67, result_path, outputDir)
    # pprint(pe.patient_pathologytype())


