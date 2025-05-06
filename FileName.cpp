#include<graphics.h>
#include<string>
#include<vector>
bool is_game_started = false;
bool running = true;
int idx_current_anim = 0;
const int PLAYER_ANIM_NUM = 6;
const int PLAYER_WIDTH = 80;
const int PLAYER_HEIGHT = 80;
const int SHADOW_WIDTH = 32;
int PLAYER_SPEED = 5;

IMAGE img_play_left[PLAYER_ANIM_NUM];
IMAGE img_play_right[PLAYER_ANIM_NUM];
POINT  player_pos = { 500,500 };
#pragma comment(lib,"MSIMG32.LIB")
#pragma comment(lib,"Winmm.lib")
class Button {
public:
	Button(RECT rect, LPCTSTR path_img_idle, LPCTSTR path_img_hovered, LPCTSTR path_img_pushed) {
		region = rect;
		loadimage(&img_idle, path_img_idle);
		loadimage(&img_hovered, path_img_hovered);
		loadimage(&img_pushed, path_img_pushed);
	}
	~Button() = default;
	void Draw() {
		switch (status) {
		case Status::Idle:
			putimage(region.left, region.top, &img_idle);
			break;
		case Status::Hovered:
			putimage(region.left, region.top, &img_hovered);
			break;
		case Status::Pushed:
			putimage(region.left, region.top, &img_pushed);
			break;
		}
	}
	void ProcessEvent(const ExMessage& msg) {
		switch (msg.message) {
		case WM_MOUSEMOVE:
			if (status == Status::Idle && CheckCursorHit(msg.x, msg.y))
				status = Status::Hovered;
			else if (status == Status::Hovered && !CheckCursorHit(msg.x, msg.y))
				status = Status::Idle;
			break;
		case WM_LBUTTONDOWN:
			if (CheckCursorHit(msg.x, msg.y))
				status = Status::Pushed;
			break;
		case WM_LBUTTONUP:
			if (status == Status::Pushed)
				OnClick();
			break;
		default:
			break;
		}
	}
protected:
	virtual void OnClick() = 0;
private:
	enum class Status {
		Idle = 0, Hovered, Pushed
	};
	RECT region;
	IMAGE img_idle;
	IMAGE img_hovered;
	IMAGE img_pushed;
	Status status = Status::Idle;
	bool CheckCursorHit(int x, int y) {
		return x > region.left && x <= region.right && y >= region.top && y <= region.bottom;
	}
};
class StartGameButton :public Button {
public:
	StartGameButton(RECT rect, LPCTSTR path_img_idle, LPCTSTR path_img_hovered, LPCTSTR path_img_pushed)
		: Button(rect,path_img_idle,path_img_hovered,path_img_pushed){}
	~StartGameButton() = default;
protected:
	void OnClick() {
		is_game_started = true;
	}
};
class QuitGameButton :public Button {
public:
	QuitGameButton(RECT rect, LPCTSTR path_img_idle, LPCTSTR path_img_hovered, LPCTSTR path_img_pushed)
		: Button(rect, path_img_idle, path_img_hovered, path_img_pushed) {}
	~QuitGameButton() = default;
protected:
	void OnClick() {
		running = false;
	}
};
inline void putimage_alpha(int x, int y, IMAGE* img) {
	int w = img->getwidth();
	int h = img->getheight();
	AlphaBlend(GetImageHDC(NULL), x, y, w, h,
		GetImageHDC(img), 0, 0, w, h, { AC_SRC_OVER,0,255,AC_SRC_ALPHA });
}
class Bullet {
public:
	POINT position = { 0,0 };
	Bullet() = default;
	~Bullet() = default;
	void Draw() const {
		setlinecolor(RGB(255, 155, 50));
		setfillcolor(RGB(200, 75, 10));
		fillcircle(position.x, position.y, RADIUS);
	}
private:
	const int RADIUS = 10;
};
/*class Animation {
private:
	int timer = 0;
	int idx_frame = 0;
	int interval_ms = 0;
	std::vector<IMAGE*>frame_list;
public:
	Animation(LPCTSTR path, int num, int interval) {
		interval_ms = interval;
		TCHAR path_file[256];
		for (size_t i = 0;i < num;i++) {
			_stprintf_s(path_file, path, i);
			IMAGE* frame = new IMAGE();
			loadimage(frame, path_file);
			frame_list.push_back(frame);
		}
	}
	~Animation() {
		for (size_t i = 0;i < frame_list.size();i++) {
			delete frame_list[i];
		}
	}
	void Play(int x, int y, int delta) {
		timer += delta;
		if (timer >= interval_ms) {
			idx_frame = (idx_frame + 1) % frame_list.size();
			timer = 0;
		}
		putimage_alpha(x, y, frame_list[idx_frame]);
	}
};*/
class EMERGY {
private:
	IMAGE img_emergy;
	int  EMERGY_SPEED = 2;
	const int EMERGY_WIDTH = 80;
	const int EMERGY_HEIGHT = 80;
	const int EMERGY_SHADOW_WIDTH = 48;
	POINT position = { 0,0 };
	bool alive = true;
public:
	EMERGY() {
		loadimage(&img_emergy, _T("img/emergy.png"));
		enum class SpawnEdge {
			UP = 0,Down,Left,Right
		};
		SpawnEdge edge = (SpawnEdge)(rand() % 4);
		switch (edge) {
		case SpawnEdge::UP:
			position.x = rand() % 1280;
			position.y = -EMERGY_HEIGHT;
			break;
		case SpawnEdge::Down:
			position.x = rand() % 1280;
			position.y = 720;
			break;
		case SpawnEdge::Left:
			position.x = -EMERGY_WIDTH;
			position.y = rand() % 720;
			break;
		case SpawnEdge::Right:
			position.x = 1280;
			position.y = rand() % 720;
			break;
		default:
			break;
		}
	}
	void Move(const POINT  player_pos) {
		int dir_x = player_pos.x - position.x;
		int dir_y = player_pos.y - position.y;
		double len_dir = sqrt(dir_x * dir_x + dir_y * dir_y);
		if (len_dir != 0) {
			double normailized_x = dir_x / len_dir;
			double normailized_y = dir_y/ len_dir;
			position.x += (int)(EMERGY_SPEED * normailized_x);
			position.y += (int)(EMERGY_SPEED * normailized_y);

		}
	}
	void Draw() {
		putimage_alpha(position.x, position.y, &img_emergy);
	}
	bool CheckPlayerCollision(const POINT player_pos){
		POINT check_position = { position.x + EMERGY_WIDTH / 2,position.y + EMERGY_HEIGHT / 2 };
		if (check_position.x > player_pos.x - 40 && check_position.x <player_pos.x + 40 && check_position.y > player_pos.y - 40 && check_position.y < player_pos.y + 40) return true;
		return false;
	}
	bool CheckBulletCollision(const Bullet& bullet) {
		POINT check_position = { position.x + EMERGY_WIDTH / 2,position.y + EMERGY_HEIGHT / 2 };
		if (check_position.x > bullet.position.x - 40 && check_position.x <bullet.position.x + 40 && check_position.y >bullet.position.y - 40 && check_position.y < bullet.position.y + 40) return true;
		return false;
	}
	void Hurt() {
		alive = false;
	}
	bool CheckAlive() {
		return alive;
	}
};
/*void LoadAnimation() {
	for (size_t i = 0;i < PLAYER_ANIM_NUM;i++) {
		std::wstring path = L"img/play_left" + std::to_wstring(i) + L".png";
		loadimage(&img_play_left[i], path.c_str());

	}
	for (size_t i = 0;i < PLAYER_ANIM_NUM;i++) {
		std::wstring path = L"img/play_right" + std::to_wstring(i) + L".png";
		loadimage(&img_play_right[i], path.c_str());

	}
}*/
void TryGenerateEnemy(std::vector<EMERGY*>& enemy_list)
{
	const int INTERVAL = 100;
	static int counter = 0;
	if ((++counter) % INTERVAL == 0) enemy_list.push_back(new EMERGY());
}
void UpdateBullets(std::vector<Bullet>& bullet_list, const POINT player_pos) {
	const double RADIAL_SPEED = 0.0045;
	const double TANGENT_SPEED = 0.0055;
	double radian_interval = 2 * 3.14159 / bullet_list.size();
	double radius = 100 + 25 * sin(GetTickCount() * RADIAL_SPEED);
	for (size_t i = 0;i < bullet_list.size();i++) {
		double radian = GetTickCount() * TANGENT_SPEED + radian_interval * i;
		bullet_list[i].position.x = player_pos.x + 40 + (int)(radius * sin(radian));
		bullet_list[i].position.y = player_pos.y + 40 + (int)(radius * cos(radian));
	}
}
void DrawPlayerScore(int score) {
	static TCHAR text[64];
	_stprintf_s(text, _T("当前玩家得分：%d"), score);
	setbkmode(TRANSPARENT);
	settextcolor(RGB(255, 85, 185));
	outtextxy(10, 10, text);
}
int main() 
{
	is_game_started = false;
	/*Animation anim_left_player(_T("img/play_left_0.png"), 6, 45);
	Animation anim_right_player(_T("img/play_left_0.png"), 6, 45);
	*/
	initgraph(1280, 720);
	std::vector<EMERGY*>enemy_list;
	std::vector<Bullet>bullet_list(3);
	RECT region_btn_start_game, region_btn_quit_game;
	region_btn_start_game.left = (1280 - 192) / 2;
	region_btn_start_game.right = (1280 - 192) / 2 + 192;
	region_btn_start_game.top = 430;
	region_btn_start_game.bottom = 430 + 75;
	region_btn_quit_game.left = (1280 - 192) / 2;
	region_btn_quit_game.right = (1280 - 192) / 2 + 192;
	region_btn_quit_game.top = 550;
	region_btn_quit_game.bottom = 550 + 75;
	StartGameButton btn_start_game = StartGameButton(region_btn_start_game,
		_T("img/ui_start_idle.png"), _T("img/ui_start_hovered.png"), _T("img/ui_start_pushed.png"));
	QuitGameButton btn_quit_game = QuitGameButton(region_btn_quit_game,
		_T("img/ui_quit_idle.png"), _T("img/ui_quit_hovered.png"), _T("img/ui_quit_pushed.png"));
	IMAGE img_menu;
	loadimage(&img_menu, _T("img/menu.png"));
	int score = 0;
	IMAGE img_shadow;
	ExMessage msg;
	IMAGE img_background;
	IMAGE img_play_left0;
	bool is_move_up = false;
	bool is_move_down = false;
	bool is_move_left = false;
	bool is_move_right = false;
	loadimage(&img_shadow,_T("img/shadow_player.png"));
	loadimage(&img_background, _T("img/background.png"));
	loadimage(&img_play_left0, _T("img/play_left_0.png"));
	BeginBatchDraw();

	while (running)
	{

		DWORD start_time = GetTickCount();
		while (peekmessage(&msg)) {
			if (is_game_started) {
				if (msg.message == WM_KEYDOWN) {
					switch (msg.vkcode) {
					case VK_UP:
						is_move_up = true;

						break;
					case VK_DOWN:
						is_move_down = true;

						break;
					case VK_LEFT:
						is_move_left = true;

						break;
					case VK_RIGHT:
						is_move_right = true;

						break;
					}
				}
				else if (msg.message == WM_KEYUP) {
					switch (msg.vkcode) {
					case VK_UP:
						is_move_up = false;

						break;
					case VK_DOWN:
						is_move_down = false;

						break;
					case VK_LEFT:
						is_move_left = false;

						break;
					case VK_RIGHT:
						is_move_right = false;

						break;
					}
				}
				if (is_move_up)player_pos.y -= PLAYER_SPEED;
				if (is_move_down)player_pos.y += PLAYER_SPEED;
				if (is_move_left)player_pos.x -= PLAYER_SPEED;
				if (is_move_right)player_pos.x += PLAYER_SPEED;
			}
			if (!is_game_started) {
				btn_start_game.ProcessEvent(msg);
				btn_quit_game.ProcessEvent(msg);
			}
		}
			if (is_game_started) {
				TryGenerateEnemy(enemy_list);
				for (EMERGY* enemy : enemy_list)
					enemy->Move(player_pos);
				for (EMERGY* enemy : enemy_list) {
					if (enemy->CheckPlayerCollision(player_pos)) {
						static TCHAR text[128];
						_stprintf_s(text, _T("最终得分：%d"), score);
						setbkmode(TRANSPARENT);
						settextcolor(RGB(255, 85, 185));
						outtextxy(100, 100, text);
						MessageBox(GetHWnd(), _T("扣‘1’看战败CG"), _T("游戏结束"), MB_OK);
						running = false;
						break;
					}
				}
				for (EMERGY* enemy : enemy_list) {
					for (const Bullet& bullet : bullet_list) {
						if (enemy->CheckBulletCollision(bullet)) {
							enemy->Hurt();
						}
					}
				}
				for (size_t i = 0;i < enemy_list.size();i++) {
					EMERGY* enemy = enemy_list[i];
					if (!enemy->CheckAlive()) {
						std::swap(enemy_list[i], enemy_list.back());
						enemy_list.pop_back();
						delete enemy;
						score++;
					}
				}
			}
			if (is_game_started) {
				cleardevice();
				putimage(0, 0, &img_background);
				int pos_shadow_x = player_pos.x + (PLAYER_WIDTH / 2 - SHADOW_WIDTH / 2);
				int pos_shadow_y = player_pos.y + PLAYER_HEIGHT - 8;
				if (player_pos.x < 0)player_pos.x = 0;
				if (player_pos.y < 0)player_pos.y = 0;
				if (player_pos.x + PLAYER_WIDTH > 1280) player_pos.x = 1280 - PLAYER_WIDTH;
				if (player_pos.y + PLAYER_HEIGHT > 720)player_pos.y = 720 - PLAYER_HEIGHT;
				putimage_alpha(player_pos.x, player_pos.y, &img_play_left0);
				putimage_alpha(pos_shadow_x, pos_shadow_y, &img_shadow);
				UpdateBullets(bullet_list, player_pos);
				for (EMERGY* enemy : enemy_list)
					enemy->Draw();
				for (const Bullet& bullet : bullet_list)
					bullet.Draw();
				DrawPlayerScore(score);
				FlushBatchDraw();
				DWORD end_time = GetTickCount();
				DWORD delta_time = end_time - start_time;
				if (delta_time < 1000 / 144)
				{
					Sleep(1000 / 144 - delta_time);
				}
			}
			if(!is_game_started) {
				cleardevice();
				putimage(0, 0, &img_menu);
				btn_start_game.Draw();
				btn_quit_game.Draw();
				FlushBatchDraw();
				DWORD end_time = GetTickCount();
				DWORD delta_time = end_time - start_time;
				if (delta_time < 1000 / 144)
				{
					Sleep(1000 / 144 - delta_time);
				}
			}
		}
	EndBatchDraw();
	return 0;
}