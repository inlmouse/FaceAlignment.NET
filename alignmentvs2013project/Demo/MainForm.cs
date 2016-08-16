using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using FacePipeline;

namespace Demo
{
    public partial class MainForm : Form
    {
        Pipeline pipeline = new Pipeline(@"GlasssixLandmarks_10stage.model");
        //Landmarks lm = new Landmarks("GlasssixLandmarks.model");
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
                Pen p = new Pen(Color.Aqua, 3);
                //ABox aBox = new ABox(0, 0, 256, 256);
                Bitmap bmp = new Bitmap(openFileDialog.FileName);
                Graphics g = Graphics.FromImage(bmp);
                //double[,]landmatks= lm.Pridect(bmp);
                Stopwatch sw = new Stopwatch();
                sw.Start();
                FrameInfo info = pipeline.GetFrameInfo(bmp);
                sw.Stop();
                ////MessageBox.Show(sw.ElapsedMilliseconds.ToString()+"\n"+info.count);
                for (int j = 0; j < info.count; j++)
                {

                    g.DrawRectangle(p, new Rectangle(info.r[j].X, info.r[j].Y, info.r[j].Width, info.r[j].Height));
                    
                    for (int i = 0; i < 68; i++)
                    {
                        //if (info.correctedlandmarks[0][i, 0] < 255 && info.correctedlandmarks[0][i, 0] > 1 &&
                        //    info.correctedlandmarks[0][i, 1] < 255 && info.correctedlandmarks[0][i, 1] > 1)
                        //{
                        bmp.SetPixel(Convert.ToInt32(info.landmarks[j][i, 0]), Convert.ToInt32(info.landmarks[j][i, 1]), Color.Crimson);
                        bmp.SetPixel(Convert.ToInt32(info.landmarks[j][i, 0]) + 1, Convert.ToInt32(info.landmarks[j][i, 1]) + 1, Color.Crimson);
                        bmp.SetPixel(Convert.ToInt32(info.landmarks[j][i, 0]) - 1, Convert.ToInt32(info.landmarks[j][i, 1]) - 1, Color.Crimson);
                        bmp.SetPixel(Convert.ToInt32(info.landmarks[j][i, 0]) + 1, Convert.ToInt32(info.landmarks[j][i, 1]) - 1, Color.Crimson);
                        bmp.SetPixel(Convert.ToInt32(info.landmarks[j][i, 0]) - 1, Convert.ToInt32(info.landmarks[j][i, 1]) + 1, Color.Crimson);
                        //}
                    }
                }



                //int count = 0;
                //for (int i = 0; i < 68; i++)
                //{
                //    if (Convert.ToInt32(landmatks[i, 0]) < 255 && Convert.ToInt32(landmatks[i, 0]) > 1 && Convert.ToInt32(landmatks[i, 1]) < 255 && Convert.ToInt32(landmatks[i, 1]) > 1)
                //    {
                //        bmp.SetPixel(Convert.ToInt32(landmatks[i, 0]), Convert.ToInt32(landmatks[i, 1]), Color.Aqua);
                //        bmp.SetPixel(Convert.ToInt32(landmatks[i, 0]) + 1, Convert.ToInt32(landmatks[i, 1]) + 1, Color.Aqua);
                //        bmp.SetPixel(Convert.ToInt32(landmatks[i, 0]) - 1, Convert.ToInt32(landmatks[i, 1]) - 1, Color.Aqua);
                //        bmp.SetPixel(Convert.ToInt32(landmatks[i, 0]) + 1, Convert.ToInt32(landmatks[i, 1]) - 1, Color.Aqua);
                //        bmp.SetPixel(Convert.ToInt32(landmatks[i, 0]) - 1, Convert.ToInt32(landmatks[i, 1]) + 1, Color.Aqua);
                //        count++;
                //    }
                //}
                FileInfo file = new FileInfo(openFileDialog.FileName);
                //Console.WriteLine(count);
                bmp.Save(@"results\" + file.Name);
                pictureBox1.Image = bmp;
                p.Dispose();
                g.Dispose();
            }
            
        }

        private void pictureBox1_Click(object sender, EventArgs e)
        {

        }

        private void MainForm_Load(object sender, EventArgs e)
        {

        }
    }
}
