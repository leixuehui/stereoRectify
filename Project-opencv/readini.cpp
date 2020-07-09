
#include <iostream>
#include <string>
#include <fstream>
#include <vector>

using namespace std;
static bool readConfigFile(const char * cfgfilepath, const string & key, vector<double>  & nums)
{
	fstream cfgFile;
	cfgFile.open(cfgfilepath);//打开文件	
	if (!cfgFile.is_open())
	{
		cout << "can not open cfg file!" << endl;
		return false;
	}

	nums.clear();
	char tmp[1000];
	while (!cfgFile.eof())//循环读取每一行
	{
		cfgFile.getline(tmp, 1000);//每行读取前1000个字符，1000个应该足够了
		string line(tmp);
		size_t pos = line.find('=');//找到每行的“=”号位置，之前是key之后是value
		if (pos == string::npos) return false;
		string tmpKey = line.substr(0, pos);//取=号之前
		string value;
		char s[30];
		
		double x;
		if (key == tmpKey)
		{
			//string x;
			////while (ss >> x)
			//cfgFile >> x;
			
			value = line.substr(pos + 1);//取=号之后
			int j = 0;
			for (int i = 0; i < value.length(); i++)
			{
				
				if (value[i] == ',' )
				{
					i++;
					j = 0;
					sscanf_s(s, "%lf", &x);  //%f对应float, %lf对应double.
					nums.push_back(x);
					memset(s, 0, sizeof(s));
				}
				s[j] = value[i];
				j++;
				if (i == value.length()-1)
				{
					sscanf_s(s, "%lf", &x);
					nums.push_back(x);
				}

			}
			return true;
		}
	}
	return false;
}
