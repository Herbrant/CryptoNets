using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using System.IO.Compression;

namespace DataPreprocess
{
    public class GetMNIST
    {
        static byte[] ReadGZFile(string fileName)
        {
            using (var fs = File.OpenRead(fileName))
            using (var gz = new GZipStream(fs, CompressionMode.Decompress))
            using (var mem = new MemoryStream())
            {
                gz.CopyTo(mem);
                return mem.ToArray();
            }
        }

        static int ReadInt32BigEndian(byte[] buf, int offset)
        {
            return (buf[offset] << 24) | (buf[offset + 1] << 16) | (buf[offset + 2] << 8) | buf[offset + 3];
        }

        static string GetImageInSparseFormat(byte[] labels, byte[,] images, int index)
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendFormat("{0}\t{1}", labels[index], 28 * 28);
            for (int j = 0; j < 28 * 28; j++)
            {
                if (images[index, j] != 0) sb.AppendFormat("\t{0}:{1}", j, images[index, j]);
            }

            return sb.ToString();
        }

        static IEnumerable<string> GetDatasetInSparseFormat(byte[] labels, byte[,] images)
        {
            for (int i = 0; i < labels.Length; i++)
                yield return GetImageInSparseFormat(labels, images, i);
        }

        public static void Run(string[] args)
        {
            const string imagesFile = "t10k-images-idx3-ubyte.gz";
            const string labelsFile = "t10k-labels-idx1-ubyte.gz";

            if (!(File.Exists(imagesFile) && File.Exists(labelsFile)))
            {
                Console.WriteLine("Please download the following files from http://yann.lecun.com/exdb/mnist/");
                Console.WriteLine("\t" + imagesFile);
                Console.WriteLine("\t" + labelsFile);
                return;
            }

            Console.WriteLine("reading input files");
            var imagesBin = ReadGZFile(imagesFile);
            var labelsBin = ReadGZFile(labelsFile);

            // verify magic numbers and read header fields (big-endian)
            if (labelsBin.Length < 8) throw new Exception("labels file too short");
            if (labelsBin[0] != 0 || labelsBin[1] != 0 || labelsBin[2] != 8 || labelsBin[3] != 1)
                throw new Exception("labels file magic number corrupted");
            int numLabels = ReadInt32BigEndian(labelsBin, 4);

            if (imagesBin.Length < 16) throw new Exception("images file too short");
            if (imagesBin[0] != 0 || imagesBin[1] != 0 || imagesBin[2] != 8 || imagesBin[3] != 3)
                throw new Exception("images file magic number corrupted");
            int numImages = ReadInt32BigEndian(imagesBin, 4);
            int numRows = ReadInt32BigEndian(imagesBin, 8);
            int numCols = ReadInt32BigEndian(imagesBin, 12);

            if (numRows != 28 || numCols != 28)
                throw new Exception($"Unexpected image size: {numRows}x{numCols}");

            if (numImages != numLabels)
                throw new Exception($"Number of images ({numImages}) does not match number of labels ({numLabels})");

            // extract labels (exact count)
            var labels = new byte[numLabels];
            Array.Copy(labelsBin, 8, labels, 0, numLabels);

            // validate image data length
            int expectedImageBytes = numImages * numRows * numCols;
            int actualImageBytes = imagesBin.Length - 16;
            if (actualImageBytes != expectedImageBytes)
                throw new Exception($"Image data length mismatch: expected {expectedImageBytes} bytes, found {actualImageBytes} bytes");

            // copy into a 2D array (or use a flat array and index)
            var images = new byte[numImages, numRows * numCols];
            Buffer.BlockCopy(imagesBin, 16, images, 0, expectedImageBytes);

            Console.WriteLine("writing MNIST-28x28-test.txt");
            File.WriteAllLines("MNIST-28x28-test.txt", GetDatasetInSparseFormat(labels, images));
            Console.WriteLine("done");
        }
    }
}
