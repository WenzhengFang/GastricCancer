import os
import shutil
import sys
import scipy.io as sio
import re
from subprocess import call


class FileIO:
    def __init__(self, fileName):
        self.fileName = fileName

    @staticmethod
    def readLines(fileName):
        """
        This function load all the lines of file, and output as 1-d list
        """
        res = []
        with open(fileName, 'r') as fin:
            res = fin.readlines()
        return res

    @staticmethod
    def readLists(fileName, sep="\t"):
        """
        This function load all the lines of file and separate the lineStr by sep,
        then, output as 2-d list
        """
        res = []
        with open(fileName, 'r') as fin:
            for line in fin:
                data = line.rstrip('\n\r').split(sep)
                res.append(data)
        return res

    @staticmethod
    def mergeFiles(fileList, outFile, header=False):
        """
        This function merge files by insert the content of subsequent files into the bottom of former file
        """
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
        """
        This function Writes all string in the list to the file by column
        """
        with open(outFile, 'w') as fout:
            for line in lines:
                fout.write(line + "\n")
        return True

    @staticmethod
    def writeLists(outFile, dataLists, sep="\t"):
        """
        This function use sep as a delimiter,
        combine the elements of two-dimensional list into a string and written to the file line by line
        """
        fout = open(outFile, 'w')
        for data in dataLists:
            data = [str(x) for x in data]
            line = sep.join(data) + "\n"
            fout.write(line)
        fout.close()

    @staticmethod
    def writeFile(outFile, str):
        """
        This function write string to the file
        """
        fout = open(outFile, 'w')
        fout.write(str)
        fout.close()

    @staticmethod
    def appendLists(inFile, dataList, sep="\t"):
        """
        This function use sep as a delimiter,
        combine the elements of two-dimensional list into a string ,
        and write to the file line by line by additional means
        """
        with open(inFile, 'a') as fopen:
            for data in dataList:
                fopen.write(sep.join(data) + "\n")
        return True

    @staticmethod
    def deleteFile(filePath):
        """
        This function delete file by file path
        """
        if os.path.exists(filePath):
            os.remove(filePath)
            return True
        else:
            print("No such file: %s!\n" % filePath)
            return False

    @staticmethod
    def copyFile(inFile, outFile):
        """
        This function copy file to target path
        """
        shutil.copy(inFile, outFile)
        return True

    @staticmethod
    def moveFile(inFile, outFile):
        """
        This function move file to target path
        """
        shutil.move(inFile, outFile)
        return True

    @staticmethod
    def existsFile(filePath):
        """
        This function judge whether the file exists
        """
        if os.path.exists(filePath):
            return True
        return False

    @staticmethod
    def renameFile(src, dst):
        """
        This function rename the file src by dst
        """
        if FileIO.existsFile(src):
            os.rename(src, dst)
            return True
        print("No such File: " + src + " found!")
        return False

    @staticmethod
    def grepFile(inFile, outFile, normPattern=[], revPattern=[]):
        """
        This function judge whether each line of the file matches the pattern,
        and write the matched lines to other file line by line
        """
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
        """
        This function make sub directory
        """
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
        """
        This function move directory by dirPath
        """
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
        """
        This function rename directory src by target path dst
        """
        if FileIO.existsFile(src):
            os.rename(src, dst)
            return True
        print("No such Directory: " + src + " found!")
        return False

    @staticmethod
    def moveDir(inDir, outDir):
        """
        This function move directory inDir to target path outDir
        """
        shutil.move(inDir, outDir)
        return True


if __name__ == "__main__":
    inFile = "tmp"
    outFile = "tmp2"
    FileIO.renameFile(inFile, outFile)