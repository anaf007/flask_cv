#coding=utf-8
"""
python2.7
"""

import pickle
from PCV.localdescriptors import sift
from numpy import sqrt

from flask import Flask,render_template,request,redirect,Blueprint
from flask_sqlalchemy import SQLAlchemy

from sqlalchemy import distinct

app = Flask(__name__)

app.config['IMG'] = 'images'

# bp = Blueprint('img',__name__,url_prefix='/img')
# print bp.endpoint
# app.register_blueprint(bp)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////tmp/test.db'
db = SQLAlchemy(app)

imlist = [
    'images/1.JPG',
    'images/2.JPG',
    'images/3.JPG',
    'images/4.JPG',
    'images/5.JPG',
    'images/6.JPG',
    'images/7.JPG',
    'images/8.JPG',
    'images/9.JPG',
]





class Imlist(db.Model):
    __tables__ = "im_list"
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(999), index=True)

    @classmethod
    def is_indexed(cls,imname):
        im = cls.query.filter_by(filename=imname).first()
        # im = self.con.execute("select rowid from imlist where filename='%s'" % imname).fetchone()
        return im != None
    
    @classmethod
    def get_id(cls,imname):
        res = cls.query.filter_by(filename=imname).first()
        if res==None: 
            imlist = Imlist()
            imlist.filename = imname
            db.session.add(imlist)
            db.session.commit()
            return imlist.id
        else:
            return res.id


class Imwords(db.Model):
    __tables__ = "im_words"
    id = db.Column(db.Integer, primary_key=True)
    im_id = db.Column(db.Integer)
    word_id = db.Column(db.Integer(), index=True)
    vocname = db.Column(db.String(999))

    @classmethod
    def add(cls,imid,word_id,vol_name):
        imwords = Imwords()
        imwords.im_id = imid
        imwords.word_id = word_id
        imwords.vol_name = vol_name
        db.session.add(imwords)
        db.session.commit()

    @classmethod
    def qdistinctid(cls, imword):
        result = cls.query.filter_by(word_id=imword).distinct().all()
        return [i.im_id for i in result]


class ImHistograms(db.Model):
    __tables__ = "im_histograms"
    id = db.Column(db.Integer, primary_key=True)
    im_id = db.Column(db.Integer)
    imhistogram = db.Column(db.String(999))
    vocname = db.Column(db.String(999))

    @classmethod
    def add(cls,imid,word_id,vol_name):
        imhistograms = ImHistograms()
        imhistograms.im_id = imid
        imhistograms.imhistogram = word_id
        imhistograms.vol_name = vol_name
        db.session.add(imhistograms)
        db.session.commit()

    @classmethod
    def get_imhistograms(cls,imname):
        im_id = Imlist.get_id(imname)
        result = cls.query.filter_by(im_id=im_id).first()
        return pickle.loads(result.imhistogram)



@app.route("/home/")
def index():

    
    global voc
    global nbr_images 
    global featlist
    nbr_images = len(imlist)
    featlist = [ imlist[i][:-3]+'sift' for i in range(nbr_images) ]
    with open('vocabulary.pkl','rb') as f:
        voc = pickle.load(f)

    if request.args.get('filename'):
        pass

    return render_template('index.html')


@app.route("/ct")
def ct():
    db.create_all()
    return "ct success"

@app.route("/add_to_index")
def add_to_index():

    for i in range(nbr_images)[:1000]:
        locs,descr = sift.read_features_from_file(featlist[i])

        if Imlist.is_indexed(imlist[i]): continue
        imid = Imlist.get_id(imlist[i])
        imwords = voc.project(descr)
        nbr_words = imwords.shape[0]
        for i in range(nbr_words):
            word = imwords[i]
            Imwords.add(imid,word,voc.name)
            # self.con.execute("insert into imwords(imid,wordid,vocname)values (?,?,?)", (imid,word,self.voc.name))
        ImHistograms.add(imid,pickle.dumps(imwords),voc.name)


    return "add_to_index"


@app.route("/printpkl")
def printpkl():
    print voc
    return 'pring pkl'


@app.route("/sdata")
def sdata():
    print Imlist.query.count()
    print Imlist.query.first().filename
    print Imlist.query.all()
    print Imwords.query.count()
    print ImHistograms.query.count()
    return 'pring pkl'


@app.route("/simg")
def simg():
    locs, descr = sift.read_features_from_file(featlist[0]) 
    iw = voc.project(descr)

    print iw

    result = candidates_from_histogram(iw)

    print  result
    return "search image"


@app.route("/query")
def quer():
    print 'imlist:',imlist[7]
    result = query(imlist[7])[:10]
    for x in result:
        print "quer:",x
    return "quer"


def candidates_from_histogram(iw):
    locs, descr = sift.read_features_from_file(featlist[0])
    iw = voc.project(descr)

    words = iw.nonzero()[0]

    # 寻找候选图像
    candidates = []
    for word in words:
        c = Imwords.qdistinctid(word)
        candidates += c

    tmp = [(w, candidates.count(w)) for w in set(candidates)]
    tmp.sort(cmp=lambda x, y: cmp(x[1], y[1]))
    tmp.reverse()
    return [w[0] for w in tmp][:10]


def query(imname):
    """ 查找所有与 imname 匹配的图像列表 """
    h = ImHistograms.get_imhistograms(imname)
    candidates = candidates_from_histogram(h)
    matchscores = []
    print 'candidates::',candidates
    for imid in candidates:
        # 获取名字
        cand_name = Imlist.query.filter_by(id=imid).first()
        if cand_name:
            cand_name = cand_name.filename
        else:
            continue
        
        cand_h = ImHistograms.get_imhistograms(cand_name)
        # 用 L2 距离度量相似性 
        cand_dist = sqrt(sum(voc.idf * (h - cand_h)/2))
        matchscores.append((cand_dist, imid,cand_name))
    # 返回排序后的距离及对应数据库 ids 列表 
    matchscores.sort()
    return matchscores

allowed_img_lambda = lambda filename: '.' in filename and filename.rsplit('.', 1)[1] in set(['jpg', 'png'])
gen_rnd_filename = lambda :"%s%s" %(datetime.datetime.now().strftime('%Y%m%d%H%M%S'), str(random.randrange(1000, 10000)))

# @csrf_protect.exempt		
@app.route('/upload',methods=['POST'])
def submit_img():
    f = request.files.get('file')
    filename = allowed_img_lambda(f.filename)
    filename = gen_rnd_filename()+'.'+f.filename.rsplit('.', 1)[1]

    dataetime = datetime.datetime.today().strftime('%Y%m%d')
    file_dir = '%s/'%('0',dataetime)
    
    if not os.path.isdir(app.config['IMG']+file_dir):
        os.makedirs(app.config['IMG']+file_dir)
    
    f.save(app.config['IMG'] +file_dir+filename)
    filename = file_dir+filename
     
    return jsonify({'success':[filename,request.form.get('id')]})
    # return redirect(url_for('.index',filename=filename))
		

