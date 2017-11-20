# Author: Fangwenzheng
# Date: 2017/11/16

import os
import shutil
import sys
import re
from subprocess import call


class FileIO:
    def __init__(self, fileName):
        self.fileName = fileName

    @staticmethod
    def readLines(fileName):
        res = []
        with open(fileName, 'r') as fin:
            res = fin.readlines()
        return res

    @staticmethod
    def readLists(fileName, sep="\t"):
        res = []
        with open(fileName, 'r') as fin:
            for line in fin:
                data = line.rstrip('\n\r').split(sep)
                res.append(data)
        return res

    @staticmethod
    def mergeFiles(fileList, outFile, header=False):
        fout = open(outFile, 'w')
        if header == True:
            fin = open(fileList[0], 'r')
            head = fin.readline()
            fout.write(head)
            fin.close()
        for file in fileList:
            fin = open(file, 'r')
            if header == True:
                head = fin.readline()
            for line in fin:
                if not line:
                    break
                fout.write(line)
            fin.close()
        fout.close()

    @staticmethod
    def writeLines(outFile, lines):
        with open(outFile, 'w') as fout:
            for line in lines:
                fout.write(line + "\n")
        return True

    @staticmethod
    def writeLists(outFile, dataLists, sep="\t"):
        fout = open(outFile, 'w')
        for data in dataLists:
            data = [str(x) for x in data]
            line = sep.join(data) + "\n"
            fout.write(line)
        fout.close()

    @staticmethod
    def writeFile(outFile, str):
        fout = open(outFile, 'w')
        fout.write(str)
        fout.close()

    @staticmethod
    def appendLists(inFile, dataList, sep="\t"):
        with open(inFile, 'a') as fopen:
            for data in dataList:
                fopen.write(sep.join(data) + "\n")
        return True

    def appendFile(self,):
        pass

    @staticmethod
    def deleteFile(filePath):
        if os.path.exists(filePath):
            os.remove(filePath)
            return True
        else:
            print("No such file: %s!\n" % filePath)
            return False

    @staticmethod
    def copyFile(inFile, outFile):
        shutil.copy(inFile, outFile)
        return True

    @staticmethod
    def moveFile(inFile, outFile):
        shutil.move(inFile, outFile)
        return True

    @staticmethod
    def existsFile(filePath):
        if os.path.exists(filePath):
            return True
        return False

    @staticmethod
    def renameFile(src, dst):
        if FileIO.existsFile(src):
            os.rename(src, dst)
            return True
        print("No such File: " + src + " found!")
        return False

    @staticmethod
    def grepFile(inFile, outFile, normPattern=[], revPattern=[]):
        if not os.path.exists(outFile):
            DirIO.mkdir(outFile[:(len(outFile) - outFile[::-1].find("/"))])
        try:
            file1 = open(inFile, "r")
            file2 = open(outFile, "w")
        except IOError:
            print("Warning, The file that you want to read does`t exists.")
            return False

        lines = file1.readlines()
        validLines = []

        for pattern in normPattern:
            if not pattern:
                continue
            for line in lines:
                if not line:
                    break
                if re.search(str(pattern), line):
                    validLines.append(line)
                else:
                    continue

            lines = validLines
            validLines = []

        for pattern in revPattern:
            if not pattern:
                continue
            for line in lines:
                if not line:
                    break
                if re.search(str(pattern), line):
                    continue
                else:
                    validLines.append(line)

            lines = validLines
            validLines = []

        file2.writelines(lines)
        file2.close()
        return True

    @staticmethod
    def sortFile_shell(fileName, args, outFile):
        cmd = "sort %s %s > %s" % (args, fileName, outFile)
        print(cmd)
        call(cmd, shell=True)


class DirIO:
    @staticmethod
    def mkdir(path):
        path = path.rstrip()
        path = path.rstrip("/")
        isExists = os.path.exists(path)
        if not isExists:
            os.makedirs(path)
            return True
        else:
            print(path + ' already exists\n')
            return False

    @staticmethod
    def removeDir(dirPath):
        if os.path.exists(dirPath):
            shutil.rmtree(dirPath)
            return True
        else:
            print("No such directory: %s!\n" % dirPath)
            return False

    @staticmethod
    def copyDir(originDirPath, newDirPath):
        # Attention!!!
        # newDirPath will be deleted before copy operation
        # make sure that no valuable data in newDirPath
        if os.path.exists(newDirPath):
            shutil.rmtree(newDirPath)
        if os.path.exists(originDirPath):
            shutil.copytree(originDirPath, newDirPath)
            return True
        else:
            print("No such directory: {0}!\n".format(originDirPath))
            return False

    @staticmethod
    def renameDir(src, dst):
        if FileIO.existsFile(src):
            os.rename(src, dst)
            return True
        print("No such Directory: " + src + " found!")
        return False

    @staticmethod
    def moveDir(inDir, outDir):
        shutil.move(inDir, outDir)
        return True


if __name__ == "__main__":
    inFile = "tmp"
    outFile = "tmp2"
    FileIO.renameFile(inFile, outFile)