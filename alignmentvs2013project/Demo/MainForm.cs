using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using alinment;

namespace Demo
{
    public partial class MainForm : Form
    {
        public MainForm()
        {
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.InitialDirectory = @"demo\";
            openFileDialog.Filter = "All file|*.*|jpg|*.jpg|png|*.png";
            openFileDialog.RestoreDirectory = true;
            openFileDialog.FilterIndex = 1;
            if (openFileDialog.ShowDialog() == DialogResult.OK)
            {
                ABox aBox = new ABox(0, 0, 256, 256);
                LbfCascade cas = new LbfCascade(@"GlasssixLandmarks.model");
                double[,] landmatks = cas.Predict(openFileDialog.FileName, aBox);
                Bitmap bmp = new Bitmap(openFileDialog.FileName);
                int count = 0;
                for (int i = 0; i < 68; i++)
                {
                    if (Convert.ToInt32(landmatks[i, 0]) < 255 && Convert.ToInt32(landmatks[i, 0]) > 1 && Convert.ToInt32(landmatks[i, 1]) < 255 && Convert.ToInt32(landmatks[i, 1])>1)
                    {
                        bmp.SetPixel(Convert.ToInt32(landmatks[i, 0]), Convert.ToInt32(landmatks[i, 1]), Color.Aqua);
                        bmp.SetPixel(Convert.ToInt32(landmatks[i, 0]) + 1, Convert.ToInt32(landmatks[i, 1]) + 1, Color.Aqua);
                        bmp.SetPixel(Convert.ToInt32(landmatks[i, 0]) - 1, Convert.ToInt32(landmatks[i, 1]) - 1, Color.Aqua);
                        bmp.SetPixel(Convert.ToInt32(landmatks[i, 0]) + 1, Convert.ToInt32(landmatks[i, 1]) - 1, Color.Aqua);
                        bmp.SetPixel(Convert.ToInt32(landmatks[i, 0]) - 1, Convert.ToInt32(landmatks[i, 1]) + 1, Color.Aqua);
                        count++;
                    }

                }
                FileInfo file = new FileInfo(openFileDialog.FileName);
                Console.WriteLine(count);
                bmp.Save(@"results\"+file.Name );
                pictureBox1.Image = bmp;
            }
            
        }
    }
}
