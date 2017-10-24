# -*- coding = utf-8 =_=
__author__ = '15624959453@163.com'

import os
import sys
import re
from glob import glob
from collections import defaultdict
from pprint import pprint

class PrioriExtract(object):
    def __init__(self, filePath):
        self.content = []
        with open(filePath, "r") as f:
            for line in f:
                self.content.append(line.strip("\r\n").split("\t"))

    # Return map table of sample name with tissue.
    def originOfSample(self):
        res = {'fresh tissue':set(), 'blood':set()}
        for info in self.content:
            res[info[8]].add(info[5])
        return res

    # Return map table of sample with patient ID.
    def patient_with_sample(self):
        res, origin_info = defaultdict(dict), self.originOfSample()
        for info in self.content:
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

    # Return all patient id in array.
    def patient_id_capture(self):
        res = []
        for line in self.content:
            if not res or line[1] != res[-1]:
                res.append(line[1])
        return res

class Preprocess(PrioriExtract):
    def __init__(self, full_info):
        PrioriExtract.__init__(self, full_info)
        self.tissue_info = self.originOfSample()
        self.patient_info = self.patient_with_sample()
        self.patient_ids = self.patient_id_capture()
        self.sample_with_path = {}

    def filePath_obtain(self, directory):
        fileList = glob(os.path.join(directory, "WETask*/SNVAndIndel/*/*.dedup.txt"))
        for filePath in fileList:
            sampleName = ".".join(os.path.basename(filePath).split(".")[:-2])
            self.sample_with_path[sampleName] = filePath
        return self.sample_with_path

    def path_from_patient(self, patientID):
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

    def filter_demo(self, patient_nums, directory, output_dir):
        self.filePath_obtain(directory)
        for id in self.patient_ids[:patient_nums]:
            samplePaths = self.path_from_patient(id)
            control_total_data = []
            exper_total_data, exper_total_file = [], os.path.join(output_dir, "patient{}_mutation.table".format(id))

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
            with open(exper_total_file, "w") as f1:
                for exper_data in exper_total_data:
                    exper_mut_site = "_".join(exper_data[:3])
                    if exper_mut_site not in control_mut_sites:
                        f1.write("\t".join(exper_data) + "\n")
                    # else:
                    #     print(exper_mut_sites)
        return True

    def filter_mutation(self, patient_nums, directory, output_dir):
        self.filePath_obtain(directory)
        mutation_file = os.path.join(output_dir, "mutation_for_1-52.table")
        single_mutation_file = os.path.join(output_dir, "mutation_single_display_for_1-52.table")
        total_mutaion_set = set()
        mutation_matrix = []

        for id in self.patient_ids[:patient_nums]:
            samplePaths = self.path_from_patient(id)
            control_total_data = []
            exper_total_data, exper_total_file = [], os.path.join(output_dir, "patient{}_mutation.table".format(id))

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
            # with open(exper_total_file, "w") as f1:
            for exper_data in exper_total_data:
                exper_mut_site = "_".join(exper_data[:3])
                if exper_mut_site not in control_mut_sites:
                    for gene in exper_data[11].split(","):
                        if gene:
                            mutation_matrix[-1].append(gene)
                            total_mutaion_set.add(gene)

        total_mutaion_list = list(total_mutaion_set)
        print(len(total_mutaion_list))
        mutation_signs = []
        for lineno, patient in enumerate(mutation_matrix):
            real_mutation, single_mutation = [], set(patient[1:])
            for mutation in total_mutaion_list:
                real_mutation.append("1" if mutation in single_mutation else "0")
            mutation_signs.append(patient[:1] + real_mutation)

        mutation_signs = [["patient id"] + total_mutaion_list] + mutation_signs
        with open(mutation_file, "w") as f1:
            for line in mutation_signs:
                f1.write("\t".join(line) + "\n")
        with open(single_mutation_file, "w") as f2:
            for line in mutation_matrix:
                line = line[:1] + list(set(line[1:]))
                f2.write("\t".join(line) + "\n")

        return True

    def snvIndel_mut_filter(self, fileName):
        sampleName, sample_tissue = ".".join(os.path.basename(fileName).split(".")[:-2]), ""
        if sampleName in self.tissue_info["blood"]:
            sample_tissue = "blood"
        else:
            sample_tissue = "fresh tissue"

        filter_data = []
        with open(fileName, "r") as f1:
            for lineno, data in enumerate(f1):
                if lineno >= 1:
                    flag = False
                    line = data.strip().split("\t")
                    line = line + ["" for _ in range(70 - len(line))]

                    if (10 < int(line[5]) + int(line[6]) and 3 <= int(line[6]) and sample_tissue == "fresh tissue") or \
                            (10 < int(line[5]) + int(line[6]) and 1 < int(line[6]) and sample_tissue == "blood"):
                        flag = True
                    if flag and \
                            float(line[12]) <= 0.05 and \
                            line[23] not in ["UNKNOWN", ""] and \
                            line[24] not in ["UNKNOWN", ""] and \
                            line[25] not in ["UNKNOWN", ""] and \
                            line[26] not in ["UNKNOWN", ""] and \
                            line[29] not in ["unknown", ""] and \
                            line[32] == "" and line[35] == "" and \
                            (line[37] == "" or (line[37] != "" and line[64] != "")):
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

    sampleInfo_path = "/lustre/users/fangwenzheng/gastricCancer/Reference/full_ver_sample_info.bed"
    result_path = "/lustre/common/WebServiceResult/Pro000068"
    outputDir = "/lustre/users/fangwenzheng/gastricCancer/result"

    quantifyFile = "/lustre/common/WebServiceResult/Pro000068/WETask17/SNVAndIndel/EGAR00001274628_FCC2C2NACXX_L3_HUMdjpXAAABAAA-91/EGAR00001274628_FCC2C2NACXX_L3_HUMdjpXAAABAAA-91.dedup.txt"
    filterFile = "/lustre/users/fangwenzheng/gastricCancer/result/filter_for_EGAR00001274628_FCC2C2NACXX_L3_HUMdjpXAAABAAA-91.table"

    pe = PrioriExtract(sampleInfo_path)
    sampleProcess = Preprocess(sampleInfo_path)
    # sampleProcess.filePath_obtain(result_path)
    # print(sampleProcess.snvIndel_mut_filter(quantifyFile))
    # print(sampleProcess.path_from_patient("211421"))
    sampleProcess.filter_mutation(52, result_path, outputDir)


